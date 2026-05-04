"""Hybrid candidate retrieval for prepare_write.

Performs candidate UNION (not vector-pool reranking):
  1. Vector pull from Chroma — top-K nearest by embedding distance
  2. BM25-only pull — top-K nearest by lexical/Okapi-BM25 over the corpus
  3. Union the two pools (dedupe by drawer_id) so a near-neighbor that
     vector misses can still be surfaced via lexical match, and vice versa
  4. Score every candidate in the union: weighted (vector × BM25) + tech boost
  5. Apply wing/room/type-aware adjustments
  6. Return blocking + advisory splits per policy thresholds

Why union (not just rerank): if the seed drawer's embedding is far from the
query's embedding (paraphrased content with limited token overlap, exactly
the M3/M6/M7 misses in the 2026-05-03 dedup test), it never enters the vector
top-K, so reranking can't recover it. A separate BM25 pull over the corpus
catches drawers with shared distinctive lexical tokens (identifiers, code
paths, technical terms) that a vector miss would silently drop.

Reuses `mempalace.searcher._bm25_scores` and `_tokenize` (zero new deps).
"""
import re
from dataclasses import dataclass

# Reuse existing BM25 implementation (zero new dependencies)
from mempalace.searcher import _bm25_scores, _tokenize


# Match common technical token shapes: dotted paths, snake_case identifiers,
# CamelCase function call patterns.
_TECH_TOKEN_RE = re.compile(
    r"\b(?:[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_.]+|[A-Z][a-zA-Z0-9]*\(|"
    r"[a-z]+_[a-z_]+|[A-Z][a-zA-Z]+\.[A-Z][a-zA-Z]+)\b"
)

# Absolute raw BM25 floor for BM25-only candidates. Below this, the "best in
# pool" candidate is just sharing common tokens, not exhibiting real lexical
# similarity. Tune via the 7-paraphrase acceptance test in test_dedup_acceptance
# if false positives or false negatives appear.
_MIN_BM25_RAW = 2.0

# Typed-memory header keys (per vault/CLAUDE.md schema). These lines (label
# AND value) must be removed entirely from the typed-memory HEADER BLOCK only —
# both label and value are boilerplate ("claude", "evidence", "confirmed" are
# shared across every drawer). Do not strip every matching lowercase line
# globally: later semantic evidence can legitimately contain lines like
# `status: 500 on checkout` or `source: webhook payload`.
_HEADER_KEYS = ("author", "type", "date", "source", "status")
_HEADER_LINE_RE = re.compile(
    r"^(?:" + "|".join(_HEADER_KEYS) + r"):.*$",
)

# Body-template label prefixes (per AAAK drawer schema). The LABEL is shared
# boilerplate across drawers — must be stripped. The VALUE after the label is
# the actual semantic content and is preserved.
#
# **Use an explicit allowlist**, not a broad "Title Case followed by colon"
# pattern. A broad pattern would strip meaningful domain prefixes like
# "HTTP:", "OAuth:", "SwiftUI:", "Postgres:", "Mailchimp:" if an agent ever
# wrote them at line start, hurting BM25 recall on exactly the distinctive
# tokens the guard needs to catch. (Codex caught this in the v1.6 review.)
#
# Allowlist drawn from:
#   - DO room: Tried because / Evidence / Caveat
#   - DONT room: Tried because / Expected / Actual / Use instead
#   - contradiction-drawer.md: Position A / Position B / Holder / Claim /
#     Context / Recommendation / Resolution status / To resolve
#   - resolution-drawer.md: Original positions / Merged stance /
#     Why this beats either position alone / Linked contradictions
_BODY_TEMPLATE_LABELS = (
    "Tried because",
    "Evidence",
    "Caveat",
    "Expected",
    "Actual",
    "Use instead",
    "Position A",
    "Position B",
    "Holder",
    "Claim",
    "Context",
    "Recommendation",
    "Resolution status",
    "To resolve",
    "Original positions",
    "Merged stance",
    "Why this beats either position alone",
    "Linked contradictions",
)
_BODY_LABEL_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(_l) for _l in _BODY_TEMPLATE_LABELS) + r"):\s*",
    re.MULTILINE,
)


def _strip_typed_header(body: str) -> str:
    """Remove template boilerplate so BM25 / distinctive-token analysis only
    sees semantic content.

    Two passes:
      1. Drop typed-memory header lines entirely (label + value) from the
         first header block only — header values like "claude", "evidence",
         "confirmed" are shared across every drawer. Semantic lines later in
         the body that happen to start with `status:` or `source:` are kept.
      2. Drop body-template label prefixes only — values are kept since
         that's where the actual semantic content lives.
    """
    title, sep1, rest = body.partition("\n\n")
    if sep1:
        header_block, sep2, semantic_body = rest.partition("\n\n")
        if sep2:
            kept_header_lines = [
                line for line in header_block.splitlines()
                if not _HEADER_LINE_RE.match(line)
            ]
            no_header = title + "\n\n" + "\n".join(kept_header_lines) + "\n\n" + semantic_body
        else:
            no_header = body
    else:
        no_header = body
    return _BODY_LABEL_RE.sub("", no_header)


@dataclass
class Candidate:
    drawer_id: str
    title: str
    body: str
    wing: str
    room: str
    type: str
    similarity: float


@dataclass
class RetrievalResult:
    blocking: list  # Candidate, similarity >= threshold
    advisory: list  # Candidate, advisory_floor <= similarity < threshold


def _extract_tech_tokens(s: str) -> set:
    return set(_TECH_TOKEN_RE.findall(s))


def _parse_drawer_type(body: str) -> str:
    """Extract the typed-memory header `type:` field from a drawer body."""
    m = re.search(r"^type:\s*(\S+)", body, re.MULTILINE)
    return m.group(1) if m else ""


def _vector_pull(col, content: str, n: int):
    """Top-N by Chroma vector similarity. Returns aligned (ids, docs, metas, distances) or empty."""
    raw = col.query(
        query_texts=[content],
        n_results=n,
        include=["metadatas", "documents", "distances"],
    )
    if not raw["ids"] or not raw["ids"][0]:
        return [], [], [], []
    return raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]


def _bm25_pull(col, content: str, n: int):
    """Top-N by BM25 over the entire palace.

    Returns (ids, docs, metas, full_raw_by_id) where:
      - ids/docs/metas are the top-N union members, sorted by raw BM25
        descending. They are the candidate-supply side of the union.
      - full_raw_by_id is `{drawer_id -> raw_bm25}` covering EVERY drawer in
        the corpus, not just the top-N. The lexical gate consumes this so
        that vector-only candidates (which won't be in the top-N) still get
        gated against full-corpus IDF rather than a union-local recompute.
        (Codex v1.8 + v2.1 reviews.)

    The top-N parallel raw-score list was dropped — no caller used it after
    full_raw_by_id became the source of truth for the gate. (Codex v2.1
    round-3 review.)

    Pulls all drawers from the collection (Chroma get with no filter), scores
    each by Okapi-BM25 against the query content, returns top-N IDs/docs/metas
    plus the full-corpus map. For palaces of ~10K drawers this is O(N)
    tokenization + scoring, ~few hundred ms. Acceptable at write time.

    BM25 runs over header-stripped content so shared typed-memory boilerplate
    (author/type/date/source/status) doesn't drive matches. Returned docs
    are the FULL original (callers may still need the header for display).
    """
    all_ = col.get(include=["documents", "metadatas"])
    if not all_["ids"]:
        return [], [], [], {}
    docs = all_["documents"]
    stripped_query = _strip_typed_header(content)
    stripped_docs = [_strip_typed_header(d) for d in docs]
    bm25_raw = _bm25_scores(stripped_query, stripped_docs)
    full_raw_by_id = dict(zip(all_["ids"], bm25_raw))
    indexed = sorted(
        range(len(docs)),
        key=lambda i: bm25_raw[i],
        reverse=True,
    )[:n]
    ids = [all_["ids"][i] for i in indexed]
    out_docs = [docs[i] for i in indexed]
    out_metas = [all_["metadatas"][i] for i in indexed]
    return ids, out_docs, out_metas, full_raw_by_id


def rank_candidates(
    *,
    content: str,
    wing: str,
    room: str,
    type_: str,
    threshold: float,
    advisory_floor: float,
    n_results: int = 20,
) -> RetrievalResult:
    """Run hybrid retrieval against the active palace; return blocking + advisory splits.

    `type_` is the type of the *incoming* write. In v1 it is captured by the
    prepare token (so commit re-check can replay the same retrieval params)
    and exposed in the API for forward compatibility, but it does NOT affect
    retrieval scoring or visibility. Type-awareness in v1 is **candidate-side
    only**: the cross-wing penalty is waived for candidate drawers whose own
    `type` is `contradiction` or `resolution` (so a contradiction-resolution
    pair stays linked across wings). A future task can add write-side
    behavior (e.g., broaden retrieval when an agent writes a
    contradiction/resolution) — that's deliberately deferred. (Codex v2.1
    review documented this.)
    """
    from mempalace import mcp_server

    col = mcp_server._get_collection()
    if not col or col.count() == 0:
        return RetrievalResult(blocking=[], advisory=[])

    # 1. Vector pull (top-K by Chroma embedding distance)
    vec_ids, vec_docs, vec_metas, vec_distances = _vector_pull(col, content, n_results)

    # 2. BM25-only pull (top-K by lexical match over the entire corpus). Also
    #    returns a full-corpus raw-score map covering every drawer in the
    #    palace — used by the lexical gate so vector-only candidates (not in
    #    the BM25 top-N) get evaluated against full-corpus IDF rather than
    #    union-local IDF.
    bm25_ids, bm25_docs, bm25_metas, bm25_full_raw_by_id = _bm25_pull(
        col, content, n_results
    )

    # 3. Union: dedupe by drawer_id, preserving first-seen ordering for stability
    union_ids: list = []
    union_docs: dict = {}
    union_metas: dict = {}
    union_distances: dict = {}  # drawer_id → distance (only set for vector hits)

    for i, did in enumerate(vec_ids):
        if did not in union_docs:
            union_ids.append(did)
            union_docs[did] = vec_docs[i]
            union_metas[did] = vec_metas[i] or {}
            union_distances[did] = vec_distances[i]
    for i, did in enumerate(bm25_ids):
        if did not in union_docs:
            union_ids.append(did)
            union_docs[did] = bm25_docs[i]
            union_metas[did] = bm25_metas[i] or {}
            # No vector distance for BM25-only candidates; treat vec_sim as 0 below

    if not union_ids:
        return RetrievalResult(blocking=[], advisory=[])

    # 4. BM25 over the union — used only for normalized scoring. The raw
    #    floor gate uses full-corpus raw scores (above), NOT this union-local
    #    recompute, since `_bm25_scores` computes IDF relative to the
    #    provided corpus.
    stripped_query = _strip_typed_header(content)
    union_stripped = [_strip_typed_header(union_docs[did]) for did in union_ids]
    bm25_union = _bm25_scores(stripped_query, union_stripped)
    max_bm25 = max(bm25_union) if bm25_union and max(bm25_union) > 0 else 1.0
    bm25_norm_by_id = {did: (bm25_union[i] / max_bm25) for i, did in enumerate(union_ids)}

    # 4b. Lexical quality gates (Codex v1.3 + v2.1 review).
    #
    # Without these gates, a query like "DO use the thing" would surface ANY
    # candidate sharing the words "do", "use", "the" with bm25_norm = 1.0
    # (because corpus-relative normalization makes the "best in pool" candidate
    # 1.0 regardless of raw signal). Two gates filter out "best in a weak pool":
    #
    #   (a) Raw BM25 floor: real lexical similarity scores raw >= ~2.0 with
    #       Lucene+ IDF over the FULL corpus; below that means just one or two
    #       common-ish tokens matched. Tunable via _MIN_BM25_RAW.
    #   (b) Distinctive-token overlap: at least one matched token must be
    #       4+ chars (filters "do", "use", "the", "is"). Aligns with how the
    #       existing _bm25_only_via_sqlite filters trigram-len tokens.
    #
    # **The gate controls TWO things, separately:**
    #   (i)  candidate visibility — BM25-only candidates are dropped if the
    #        gate fails (they have no other signal to carry them).
    #   (ii) BM25 score contribution — a candidate's normalized BM25 only
    #        feeds into scoring if the gate passes. This applies to vector
    #        hits too: a weak-vec candidate with spurious common-token overlap
    #        must not be promoted to blocking via the union-relative bm25_norm
    #        when its full-corpus raw BM25 is below _MIN_BM25_RAW. (Codex v2.1
    #        review caught the original bypass for vector hits.)
    #
    # Distinctive tokens computed over header-stripped query — boilerplate
    # tokens like "claude", "evidence", "confirmed" must not count as
    # distinctive overlap.
    distinctive_query_tokens = {t for t in _tokenize(stripped_query) if len(t) >= 4}
    bm25_signal_passes_gate = {}  # gate decision for BM25 score contribution
    for did in union_ids:
        # Use full-corpus raw BM25 for ALL union IDs (no fallback to union-
        # local). A vector-only candidate that didn't make the BM25 top-N
        # still has a real full-corpus score in `bm25_full_raw_by_id`. The
        # 0.0 fallback is defensive only — every union id should have come
        # from either vec_ids or bm25_ids, both of which are subsets of the
        # collection that bm25_full_raw_by_id covers.
        raw = bm25_full_raw_by_id.get(did, 0.0)
        if raw < _MIN_BM25_RAW:
            bm25_signal_passes_gate[did] = False
            continue
        # Candidate tokens also computed over header-stripped body
        cand_tokens = set(_tokenize(_strip_typed_header(union_docs[did])))
        if not (distinctive_query_tokens & cand_tokens):
            bm25_signal_passes_gate[did] = False
            continue
        bm25_signal_passes_gate[did] = True

    # 5. Identifier-token boost (write-guard specific)
    query_techs = _extract_tech_tokens(content)
    tech_boost_by_id = {}
    for did in union_ids:
        doc_techs = _extract_tech_tokens(union_docs[did])
        overlap = query_techs & doc_techs
        tech_boost_by_id[did] = 0.05 * min(len(overlap), 3)  # cap boost at 0.15

    # 6. Combine + wing/room/type-aware adjustment
    #
    # Score formula: max-primary + small-secondary-boost, NOT weighted sum.
    # Reason: with weighted sum (0.6*vec + 0.3*bm25), a BM25-only candidate
    # (vec_sim=0) caps at ~0.48, below the default advisory floor (0.5) — so
    # the BM25 union becomes mathematically inert. The max-primary form lets
    # EITHER signal alone hit the blocking threshold, with a small lift when
    # both fire. (Codex caught this in the v1.2 review.)
    #
    # Visibility vs BM25 contribution (Codex v1.3 + v2.1 review):
    #   - BM25-only candidates that fail the lexical gate are dropped entirely
    #     (no other signal carries them).
    #   - Vector hits stay visible regardless (vector signal carries them),
    #     but their BM25 contribution is zeroed when the lexical gate fails.
    #     This prevents a weak-vec candidate from being promoted to blocking
    #     via union-relative bm25_norm when full-corpus raw BM25 is below
    #     _MIN_BM25_RAW.
    candidates = []
    for did in union_ids:
        is_vector_hit = did in union_distances
        bm25_allowed = bm25_signal_passes_gate.get(did, False)

        # BM25-only candidate that failed the gate: drop entirely.
        if not is_vector_hit and not bm25_allowed:
            continue

        meta = union_metas[did]
        cand_wing = meta.get("wing", "")
        cand_room = meta.get("room", "")
        body = union_docs[did]
        cand_type = _parse_drawer_type(body)

        # vec_sim: from distance if vector pool found it; else 0 (BM25-only candidate)
        if is_vector_hit:
            vec_sim = max(0.0, 1 - union_distances[did])
        else:
            vec_sim = 0.0

        # BM25 contributes to scoring ONLY if the lexical gate passes.
        # Applies to vector hits too — see "Visibility vs BM25 contribution"
        # above.
        bm25 = bm25_norm_by_id[did] if bm25_allowed else 0.0

        # Max-primary scoring: either signal alone can be sufficient;
        # both firing gives a small additive lift.
        primary = max(vec_sim, bm25)
        secondary = min(vec_sim, bm25)
        score = primary + 0.15 * secondary + tech_boost_by_id[did]

        # Wing/room/type-aware adjustment
        if cand_wing != wing:
            if cand_type not in ("contradiction", "resolution"):
                score -= 0.05
        elif cand_room == room:
            score += 0.03

        score = max(0.0, min(1.0, score))
        title = body.split("\n", 1)[0][:200]

        candidates.append(Candidate(
            drawer_id=did,
            title=title,
            body=body,
            wing=cand_wing,
            room=cand_room,
            type=cand_type,
            similarity=round(score, 3),
        ))

    candidates.sort(key=lambda c: c.similarity, reverse=True)

    blocking = [c for c in candidates if c.similarity >= threshold]
    advisory = [c for c in candidates if advisory_floor <= c.similarity < threshold]

    return RetrievalResult(blocking=blocking, advisory=advisory)
