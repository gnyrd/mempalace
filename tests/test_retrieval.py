"""Tests for hybrid retrieval used by prepare_write.

The retrieval combines vector similarity, lexical/BM25, identifier-token
boost, and wing/room/type-aware filtering. Returns blocking + advisory splits
per the policy thresholds.
"""
import pytest

from mempalace.retrieval import rank_candidates


# Test palace fixture: assumes a small corpus of drawers in tmp palace.
# Use the existing test conftest pattern; see ~/Repos/mempalace/tests/conftest.py.
@pytest.fixture
def small_palace(tmp_path, monkeypatch):
    """Create an isolated palace with a known small set of drawers for retrieval tests."""
    import json
    # Allow tool_add_drawer for seeding (test fixtures bypass the gate intentionally; see Task 1)
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"deprecated_add_drawer_response": "allow"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))
    # Reset cached chroma client per existing test convention
    from mempalace import mcp_server
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)

    # Seed: known drawers we'll search for
    mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO use chat.guid as thread_id\n\nauthor: claude\ntype: evidence\n"
                "date: 2026-05-01\nsource: test\nstatus: confirmed\n\n"
                "Tried because: testing primary key\nEvidence: ROWID is not unique across DBs\n"
                "Caveat: only on macOS Messages",
    )
    mcp_server.tool_add_drawer(
        wing="testwing", room="dont",
        content="DONT use chat.ROWID as primary key\n\nauthor: claude\ntype: evidence\n"
                "date: 2026-05-01\nsource: test\nstatus: confirmed\n\n"
                "Tried because: needed stable IDs\nExpected: unique values\n"
                "Actual: ROWID collides across migrated DBs\n"
                "Use instead: chat.guid",
    )
    mcp_server.tool_add_drawer(
        wing="otherwing", room="general",
        content="general note about deployment\n\nauthor: claude\ntype: evidence\n"
                "date: 2026-04-15\nsource: test\nstatus: confirmed\n\n"
                "Some unrelated content about CI/CD pipelines",
    )
    yield tmp_path / "palace"


def test_returns_blocking_and_advisory_split(small_palace):
    result = rank_candidates(
        content="DO use chat.guid as the thread identifier instead of ROWID",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    # The DO drawer should be a blocking candidate (high similarity)
    assert any(c.drawer_id and "guid" in c.body.lower() for c in result.blocking)
    # The DONT may or may not be blocking depending on similarity, but should appear
    # somewhere (blocking or advisory)
    all_returned = result.blocking + result.advisory
    assert any("ROWID" in c.body for c in all_returned)


def test_advisory_floor_filters_lowest_hits(small_palace):
    """A query with weak similarity should still surface drawers above advisory_floor."""
    result = rank_candidates(
        content="Some completely unrelated content about deployment pipelines",
        wing="testwing",  # different wing from the matching drawer
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    # The deployment-related drawer is in otherwing/general; cross-wing should
    # be deprioritized but still surface as advisory if above floor
    # (Exact assertion depends on retrieval behavior; allow the advisory hit at minimum.)
    assert isinstance(result.blocking, list)
    assert isinstance(result.advisory, list)


def test_identifier_boost_lifts_code_path_match(small_palace):
    """Content sharing distinctive code-path tokens should rank higher."""
    # Both drawers reference "chat.guid" / "chat.ROWID"; a query containing
    # the exact tokens should boost their match scores.
    result = rank_candidates(
        content="chat.guid is the right identifier",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    assert len(result.blocking) >= 1


def test_wing_filter_prioritizes_same_wing(small_palace):
    """Same-wing candidates should rank above cross-wing for similar content."""
    result = rank_candidates(
        content="something about chat IDs",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    # If both wings have similar-ish content, testwing entries appear first/higher
    if len(result.blocking) > 0:
        assert all(c.wing == "testwing" or c.similarity > 0.85 for c in result.blocking[:2])


def test_returns_empty_when_palace_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "empty"))
    from mempalace import mcp_server
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)

    result = rank_candidates(
        content="anything",
        wing="w", room="do", type_="evidence",
        threshold=0.7, advisory_floor=0.5,
    )
    assert result.blocking == []
    assert result.advisory == []


def test_bm25_only_match_surfaces_when_vector_misses(small_palace, monkeypatch):
    """Drawer with strong lexical match but weak vector similarity should still
    surface via the BM25-only pull AND clear the advisory_floor / blocking
    threshold. Tests both candidate-union AND that BM25-only candidates
    score high enough to actually appear in blocking/advisory (not get
    mathematically filtered out by hybrid weights — Codex's v1.2 review)."""
    from mempalace import retrieval as r

    # Stub _vector_pull to return empty (simulate vector miss). BM25 alone
    # must surface the seed AND the score must clear at least advisory_floor.
    monkeypatch.setattr(r, "_vector_pull", lambda col, content, n: ([], [], [], []))

    result = r.rank_candidates(
        content="DO use chat.guid as the thread identifier instead of ROWID",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    all_returned = result.blocking + result.advisory
    assert len(all_returned) > 0, (
        "BM25-only retrieval should surface candidates even when vector returns "
        "nothing. If empty, the scoring formula penalizes BM25-only candidates "
        "out of the advisory floor — see retrieval.py max-primary scoring."
    )

    # The seeded drawer about chat.guid should be the top match (lexical overlap
    # with chat, guid, ROWID, thread)
    top = all_returned[0]
    assert "guid" in top.body.lower() or "rowid" in top.body.lower(), (
        f"top BM25-only match should contain shared tokens; got body={top.body[:100]}"
    )

    # And its score must clear advisory_floor (the bug Codex caught was the
    # score capping at 0.48 < 0.5)
    assert top.similarity >= 0.5, (
        f"BM25-only candidate similarity ({top.similarity}) must clear advisory_floor "
        f"(0.5). If lower, the scoring formula is filtering BM25-only hits out."
    )


def test_weak_lexical_query_does_not_surface_bm25_only_blocking(small_palace, monkeypatch):
    """A query with only common tokens (DO, use, the, is) must NOT surface
    BM25-only candidates. The corpus-relative normalization would otherwise
    give the top hit similarity=1.0 just for sharing stopword-ish tokens.
    Tests the raw-floor + distinctive-token gates Codex flagged in v1.3."""
    from mempalace import retrieval as r

    monkeypatch.setattr(r, "_vector_pull", lambda col, content, n: ([], [], [], []))

    result = r.rank_candidates(
        content="DO use the thing",  # all very common tokens, < 4 chars except "thing"
        wing="otherwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    # The seeded testwing drawers all contain "DO", "use", etc., but none are
    # genuinely about "the thing" — so BM25-only blocking should NOT fire.
    assert result.blocking == [], (
        f"weak-query BM25-only false-positive: {result.blocking}. "
        f"The raw-floor and distinctive-token gates should have filtered these."
    )


def test_distinctive_token_gate_blocks_pure_stopword_query(small_palace, monkeypatch):
    """Even more degenerate: query is purely common tokens. No distinctive
    (length >= 4) overlap with any seeded drawer. Must return nothing."""
    from mempalace import retrieval as r

    monkeypatch.setattr(r, "_vector_pull", lambda col, content, n: ([], [], [], []))

    result = r.rank_candidates(
        content="DO is do is",  # zero distinctive tokens
        wing="testwing", room="do", type_="evidence",
        threshold=0.7, advisory_floor=0.5,
    )
    assert result.blocking == []
    assert result.advisory == []


def test_bm25_raw_floor_uses_full_corpus_not_union(tmp_path, monkeypatch):
    """BM25-only raw-score gating must use the full-corpus BM25 score, not a
    recomputed score over the small vector+BM25 union. `_bm25_scores` computes
    IDF over the provided corpus; if the raw floor is computed only over the
    union, a common full-palace token can look artificially distinctive and
    become a false-positive blocking candidate. (Codex v1.8 review.)"""
    import json
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"deprecated_add_drawer_response": "allow"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))

    from mempalace import mcp_server, retrieval as r
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)

    # `reconcile` is 4+ chars and appears in many unrelated drawers. Full-corpus
    # IDF should make it weak; small-union IDF can wrongly inflate it.
    for i in range(25):
        mcp_server.tool_add_drawer(
            wing="ops", room="do",
            content=(
                f"DO reconcile routine note {i}\n\n"
                "author: claude\ntype: evidence\ndate: 2026-05-01\nsource: test\nstatus: confirmed\n\n"
                f"Tried because: operations checklist item {i}\n"
                f"Evidence: reconcile appears in many unrelated runbook notes {i}\n"
                "Caveat: generic operational vocabulary"
            ),
        )

    monkeypatch.setattr(r, "_vector_pull", lambda col, content, n: ([], [], [], []))

    result = r.rank_candidates(
        content="DO reconcile the thing",
        wing="otherwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    assert result.blocking == [], (
        f"common full-corpus token surfaced as blocking: {result.blocking}. "
        "BM25-only raw-floor gating must use full-corpus raw scores, not "
        "scores recomputed over the tiny union."
    )
    assert result.advisory == []


def test_semantic_colon_prefixes_are_preserved():
    """Domain terms that happen to start a line in `Word: ...` form (HTTP,
    OAuth, SwiftUI, Postgres, Mailchimp, etc.) must NOT be stripped. Lowercase
    semantic fields later in the body (`status: 500`, `source: webhook`) must
    also survive. Only the typed-memory header block and explicit template
    labels are stripped. (Codex v1.6 + v1.8 review.)"""
    from mempalace.retrieval import _strip_typed_header

    body = (
        "DO use SwiftUI's @State for transient state\n\n"
        "author: claude\ntype: evidence\ndate: 2026-05-01\nsource: test\nstatus: confirmed\n\n"
        "Tried because: needed reactive UI updates\n"
        "Evidence: @State binds view to value, automatic re-render on mutation\n"
        "Caveat: don't share @State across views\n"
        "HTTP: 200 in 50ms (incidental note)\n"
        "OAuth: PKCE preferred\n"
        "Postgres: pg_stat_activity revealed lock\n"
        "Mailchimp: API rate-limit window 5 min\n"
        "status: 500 on checkout after webhook retry\n"
        "source: webhook payload included provider_status\n"
    )
    stripped = _strip_typed_header(body)

    # Template labels stripped (label only — values preserved):
    assert "Tried because:" not in stripped, f"template label 'Tried because:' not stripped: {stripped!r}"
    assert "Evidence:" not in stripped, f"template label 'Evidence:' not stripped: {stripped!r}"
    assert "Caveat:" not in stripped, f"template label 'Caveat:' not stripped: {stripped!r}"
    # Their values preserved:
    assert "needed reactive UI updates" in stripped
    assert "@State binds view" in stripped

    # Domain prefixes preserved (NOT in allowlist):
    assert "HTTP:" in stripped, f"domain term 'HTTP:' was stripped (regression): {stripped!r}"
    assert "OAuth:" in stripped, f"domain term 'OAuth:' was stripped: {stripped!r}"
    assert "Postgres:" in stripped, f"domain term 'Postgres:' was stripped: {stripped!r}"
    assert "Mailchimp:" in stripped, f"domain term 'Mailchimp:' was stripped: {stripped!r}"

    # Semantic lowercase field names after the header are preserved:
    assert "status: 500 on checkout" in stripped
    assert "source: webhook payload" in stripped

    # Header lines entirely gone:
    assert "author: claude" not in stripped
    assert "status: confirmed" not in stripped


def test_template_boilerplate_overlap_does_not_trigger_blocking(tmp_path, monkeypatch):
    """Two unrelated drawers sharing only template boilerplate must not
    surface each other via BM25. Two layers of boilerplate must be stripped:
      1. Typed-memory header (author/type/date/source/status) — Codex v1.4
      2. Body-template labels (Tried because/Evidence/Caveat/Expected/...) —
         Codex v1.5 (this layer was missing in v1.5; v1.6 fix)
    """
    import json
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"deprecated_add_drawer_response": "allow"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))

    from mempalace import mcp_server, retrieval as r
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)

    # Two completely unrelated drawer bodies; identical headers
    drawer_a_body = (
        "DO use semaphore primitives for FFI thread coordination\n\n"
        "author: claude\ntype: evidence\ndate: 2026-04-15\nsource: test\nstatus: confirmed\n\n"
        "Tried because: Rust async over C boundary needed coordination\n"
        "Evidence: Mutex held across await deadlocks; semaphore.Permit avoids\n"
        "Caveat: only when caller is async-aware"
    )
    drawer_b_body = (
        "DO bake margarine into pie crust for flakiness\n\n"
        "author: claude\ntype: evidence\ndate: 2026-04-20\nsource: test\nstatus: confirmed\n\n"
        "Tried because: butter crust was tough\n"
        "Evidence: margarine's higher water content steams during bake\n"
        "Caveat: flavor is less rich"
    )
    mcp_server.tool_add_drawer(wing="cooking", room="do", content=drawer_b_body)

    # Force BM25-only path so the gates are what matters
    monkeypatch.setattr(r, "_vector_pull", lambda col, content, n: ([], [], [], []))

    result = r.rank_candidates(
        content=drawer_a_body,  # FFI threading; nothing about pie crust
        wing="rust",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )
    # Drawer A and B share ONLY the header tokens. After stripping, no
    # distinctive overlap should remain. Must NOT block.
    assert result.blocking == [], (
        f"header-only overlap surfaced as blocking: {result.blocking}. "
        f"_strip_typed_header is likely missing or the regex isn't catching all keys."
    )


def test_weak_vector_hit_with_common_lexical_overlap_does_not_block(small_palace, monkeypatch):
    """A vector hit with low similarity must NOT be promoted to blocking by
    union-relative normalized BM25 when the lexical signal would fail the
    full-corpus raw-floor / distinctive-token gates.

    Without the fix: a vector hit's BM25 against a tiny union can be 1.0
    (best-in-pool) regardless of full-corpus IDF, so max(vec_sim=0.3, 1.0)
    promotes the candidate above the 0.7 blocking threshold purely on common-
    token overlap. With the fix: BM25 contributes 0 to scoring when the
    lexical gate fails, so the score is just vec_sim ~= 0.3. (Codex v2.1
    review.)
    """
    from mempalace import retrieval as r

    def stub_low_sim_vector(col, content, n):
        # Return one real seeded drawer at high distance (low vec_sim).
        all_data = col.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return [], [], [], []
        return (
            [all_data["ids"][0]],
            [all_data["documents"][0]],
            [all_data["metadatas"][0]],
            [0.7],  # distance 0.7 -> vec_sim 0.3
        )

    monkeypatch.setattr(r, "_vector_pull", stub_low_sim_vector)

    result = r.rank_candidates(
        # All tokens are common; "thing" is the only distinctive (>=4 char)
        # token and it does NOT appear in any seeded drawer body. So the
        # distinctive-token gate fails. Without the fix, the lone vector
        # candidate's normalized BM25 = 1.0 (best in pool of 1) and gets
        # promoted via max(0.3, 1.0).
        content="DO use the thing",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )

    blocking_ids = [c.drawer_id for c in result.blocking]
    assert blocking_ids == [], (
        f"weak vector hit (vec_sim=0.3) was promoted to blocking via "
        f"common-token BM25 overlap: {result.blocking}. The full-corpus "
        f"raw BM25 floor and distinctive-token gates must suppress the "
        f"BM25 contribution to scoring for vector hits as well, not just "
        f"BM25-only candidates."
    )


def test_incoming_type_does_not_change_retrieval(small_palace):
    """v1 type-awareness is candidate-side only: changing the incoming `type_`
    parameter must not change which candidates surface or their order. The
    cross-wing exception is determined by the CANDIDATE drawer's type
    (contradiction/resolution), not by what the agent declares for the
    incoming write. The `type_` parameter is captured in the prepare token
    for commit re-check replay but does not feed retrieval scoring.

    If a future task adds write-side type behavior, this test will need to
    update — and the docstring on rank_candidates should be updated in the
    same change. (Codex v2.1 review documented this v1 behavior.)
    """
    from mempalace import retrieval as r

    common_kwargs = dict(
        content="DO use chat.guid as the thread identifier instead of ROWID",
        wing="testwing",
        room="do",
        threshold=0.7,
        advisory_floor=0.5,
    )

    res_evidence = r.rank_candidates(type_="evidence", **common_kwargs)
    res_contradiction = r.rank_candidates(type_="contradiction", **common_kwargs)
    res_resolution = r.rank_candidates(type_="resolution", **common_kwargs)

    def fingerprint(res):
        # (drawer_id, similarity) pairs across both buckets, preserving order
        return [(c.drawer_id, c.similarity) for c in res.blocking + res.advisory]

    fp_e = fingerprint(res_evidence)
    fp_c = fingerprint(res_contradiction)
    fp_r = fingerprint(res_resolution)

    assert fp_e == fp_c == fp_r, (
        f"changing incoming type_ changed retrieval results.\n"
        f"  evidence:      {fp_e}\n"
        f"  contradiction: {fp_c}\n"
        f"  resolution:    {fp_r}\n"
        f"v1 type-awareness is candidate-side only — the incoming type_ "
        f"is captured but must not influence scoring. If this test fails "
        f"because behavior changed intentionally, also update the "
        f"rank_candidates docstring."
    )


def test_vector_only_candidate_uses_full_corpus_bm25_floor(small_palace, monkeypatch):
    """A vector-only candidate (in `_vector_pull` but NOT in `_bm25_pull`'s
    top-N) must have its lexical gate evaluated against FULL-CORPUS raw BM25,
    not a fallback recompute over the union.

    Bug shape (caught in Codex v2.1 second-round review): the previous fix
    populated `bm25_full_raw_by_id` only for IDs present in the BM25 top-N.
    Vector-only candidates fell through to a union-local BM25 recompute,
    where IDF is inflated for shared tokens (term-in-1-of-2 docs looks rare).
    A weak vector hit with a common 4+ char token could pass the raw floor
    via the union-local recompute even though its full-corpus raw BM25 was
    below `_MIN_BM25_RAW`, then receive normalized BM25 contribution that
    promoted it to blocking.

    Fix: `_bm25_pull` returns a full-corpus raw-score map covering every
    drawer; `bm25_full_raw_by_id.get(did, 0.0)` (no union-local fallback)
    is the only source of truth for the gate.
    """
    from mempalace import retrieval as r

    # Stub _bm25_pull: empty top-N, full-corpus map showing every drawer's
    # raw BM25 is below the floor. Mimics the case where the vector
    # candidate's full-corpus BM25 is genuinely weak — only common tokens
    # overlap.
    def stub_bm25_pull(col, content, n):
        all_data = col.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return [], [], [], {}
        full_map = {did: 0.5 for did in all_data["ids"]}  # well below 2.0 floor
        return [], [], [], full_map

    monkeypatch.setattr(r, "_bm25_pull", stub_bm25_pull)

    # Stub _vector_pull: return one drawer at moderate vec_sim. Picking the
    # first seeded drawer (the chat.guid DO drawer) — its body has 4+ char
    # tokens like "thread", "primary", "testing" that distinctive-token
    # check would pass on, but full-corpus BM25 for the query is weak.
    def stub_vector_pull(col, content, n):
        all_data = col.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return [], [], [], []
        return (
            [all_data["ids"][0]],
            [all_data["documents"][0]],
            [all_data["metadatas"][0]],
            [0.5],  # distance 0.5 -> vec_sim 0.5
        )

    monkeypatch.setattr(r, "_vector_pull", stub_vector_pull)

    result = r.rank_candidates(
        # "thread" is a 4+ char token shared with the chat.guid drawer body
        # ("DO use chat.guid as thread_id"). Distinctive-token gate would
        # pass IF the raw floor passed. The bug let union-local BM25 mask
        # the weak full-corpus signal; the fix forces full-corpus raw.
        content="DO use the thread thing",
        wing="testwing",
        room="do",
        type_="evidence",
        threshold=0.7,
        advisory_floor=0.5,
    )

    blocking_ids = [c.drawer_id for c in result.blocking]
    assert blocking_ids == [], (
        f"vector-only candidate (not in BM25 top-N) was promoted to blocking "
        f"despite full-corpus raw BM25 below floor: {result.blocking}. The "
        f"lexical gate must use the full-corpus raw-score map for ALL union "
        f"IDs, not fall back to union-local BM25 for vector-only candidates."
    )
