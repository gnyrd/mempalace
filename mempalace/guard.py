"""Write-guard core.

Owns the prepare/commit state machine: token issuance, fate-vs-relation
validation, per-relation operations (calling existing tool_create_tunnel /
tool_update_drawer), preimage capture, compensating rollback, audit-trail
builder.

This file is intentionally a single module to keep related state machinery
co-located. Sub-sections (tokens, validation, operations, atomicity) are
clearly separated for readability.
"""
import secrets
import time
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Token state
# ============================================================================


class TokenError(Exception):
    """Raised when a guard token is unknown, expired, or otherwise invalid."""


@dataclass
class TokenRecord:
    content_hash: str
    blocking_ids: list
    advisory_ids: list
    wing: str
    room: str
    type: str
    prepared_content: str
    # Retrieval params captured at prepare time, replayed at commit re-check
    # so policy changes between prepare and commit don't produce false
    # candidates_changed errors.
    effective_threshold: float
    effective_advisory_floor: float
    issued_at: float
    expires_at: float


class TokenStore:
    """In-memory store for guard tokens. Process-local; lost on server restart.

    Server restart wipes all tokens. Acceptable: agent gets ``token_expired_or_unknown``
    on commit, calls prepare_write again. No correctness impact; recovery is a
    single retry. (See spec § Guard token.)
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._tokens: dict = {}

    def issue(
        self,
        *,
        content_hash: str,
        blocking_ids: list,
        advisory_ids: list,
        wing: str,
        room: str,
        type: str,
        prepared_content: str,
        effective_threshold: float,
        effective_advisory_floor: float,
    ) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._tokens[token] = TokenRecord(
            content_hash=content_hash,
            blocking_ids=list(blocking_ids),
            advisory_ids=list(advisory_ids),
            wing=wing,
            room=room,
            type=type,
            prepared_content=prepared_content,
            effective_threshold=effective_threshold,
            effective_advisory_floor=effective_advisory_floor,
            issued_at=now,
            expires_at=now + self._ttl,
        )
        return token

    def get(self, token: str) -> TokenRecord:
        rec = self._tokens.get(token)
        if rec is None:
            raise TokenError(f"unknown token: {token[:8]}...")
        if time.time() > rec.expires_at:
            self._tokens.pop(token, None)
            raise TokenError(f"expired token: {token[:8]}...")
        return rec

    def consume(self, token: str) -> TokenRecord:
        """Get + remove. Tokens are single-use to prevent replay."""
        rec = self.get(token)
        self._tokens.pop(token, None)
        return rec

    def purge_expired(self) -> int:
        now = time.time()
        expired = [t for t, rec in self._tokens.items() if now > rec.expires_at]
        for t in expired:
            self._tokens.pop(t, None)
        return len(expired)

    def size(self) -> int:
        return len(self._tokens)


# ============================================================================
# Fate / Relation enums
# ============================================================================


class Fate:
    CREATE_NEW = "create_new"
    SKIP_NEW = "skip_new"
    UPDATE_EXISTING = "update_existing"

    ALL = (CREATE_NEW, SKIP_NEW, UPDATE_EXISTING)


class Relation:
    DUPLICATE = "duplicate"
    SUPERSEDES = "supersedes"
    RESOLVES_CONTRADICTION = "resolves_contradiction"
    RELATED_DISTINCT = "related_distinct"
    STRENGTHENS_TARGET = "strengthens_target"
    IGNORED_SPURIOUS = "ignored_spurious"

    ALL = (
        DUPLICATE,
        SUPERSEDES,
        RESOLVES_CONTRADICTION,
        RELATED_DISTINCT,
        STRENGTHENS_TARGET,
        IGNORED_SPURIOUS,
    )

    # Relations that imply a new drawer is being created
    CREATING = (SUPERSEDES, RESOLVES_CONTRADICTION, RELATED_DISTINCT, IGNORED_SPURIOUS)
    # Relations that imply NO new drawer
    NON_CREATING = (DUPLICATE, STRENGTHENS_TARGET)


# ============================================================================
# Validation
# ============================================================================


class ValidationError(ValueError):
    """Raised when commit_write arguments fail fate/relation validation."""


def validate_commit_args(
    *,
    fate: str,
    candidate_relations: list,
    blocking_ids: list,
    merged_content: Optional[str],
) -> None:
    """Validate fate + candidate_relations + merged_content per the spec.

    Raises ValidationError on any rule violation. Returns None on success.
    """
    if fate not in Fate.ALL:
        raise ValidationError(f"unknown fate: {fate!r}")

    # Shape: each candidate_relations entry must be a dict with drawer_id +
    # relation. Without this, malformed entries surface as bare KeyError to
    # MCP callers instead of typed ValidationError. (Codex v2.1 review.)
    for i, r in enumerate(candidate_relations):
        if not isinstance(r, dict):
            raise ValidationError(
                f"candidate_relations[{i}] must be a dict, got {type(r).__name__}: {r!r}"
            )
        if "drawer_id" not in r:
            raise ValidationError(
                f"candidate_relations[{i}] missing required key 'drawer_id': {r!r}"
            )
        if "relation" not in r:
            raise ValidationError(
                f"candidate_relations[{i}] missing required key 'relation': {r!r}"
            )

    # Coverage: every blocking candidate addressed exactly once
    addressed = [r["drawer_id"] for r in candidate_relations]
    blocking = list(blocking_ids)
    seen = set()
    for did in addressed:
        if did in seen:
            raise ValidationError(f"candidate {did!r} addressed twice")
        seen.add(did)
        if did not in blocking:
            raise ValidationError(f"candidate {did!r} not in blocking_candidates")
    missing = [b for b in blocking if b not in seen]
    if missing:
        raise ValidationError(f"blocking candidates not addressed: {missing}")

    # Per-relation shape: related_distinct requires association
    for r in candidate_relations:
        rel = r["relation"]
        if rel not in Relation.ALL:
            raise ValidationError(f"unknown relation: {rel!r}")
        if rel == Relation.RELATED_DISTINCT and not r.get("association"):
            raise ValidationError(
                f"relation 'related_distinct' on {r['drawer_id']!r} requires association"
            )

    relations_used = {r["relation"] for r in candidate_relations}

    # Fate-vs-relation consistency
    if fate == Fate.CREATE_NEW:
        bad = relations_used & set(Relation.NON_CREATING)
        if bad:
            for b in bad:
                raise ValidationError(
                    f"relation {b!r} not allowed with fate=create_new "
                    f"(it implies no new drawer); use fate=skip_new or update_existing"
                )

    elif fate == Fate.SKIP_NEW:
        if Relation.DUPLICATE not in relations_used:
            raise ValidationError(
                "fate=skip_new requires at least one relation=duplicate"
            )
        allowed = {Relation.DUPLICATE, Relation.IGNORED_SPURIOUS}
        if relations_used - allowed:
            raise ValidationError(
                "fate=skip_new permits only relations in "
                "{duplicate, ignored_spurious}"
            )
        if merged_content is not None:
            raise ValidationError("merged_content must be absent for fate=skip_new")

    elif fate == Fate.UPDATE_EXISTING:
        strengthens = [r for r in candidate_relations if r["relation"] == Relation.STRENGTHENS_TARGET]
        if len(strengthens) != 1:
            raise ValidationError(
                "fate=update_existing requires exactly one strengthens_target relation"
            )
        allowed = {Relation.STRENGTHENS_TARGET, Relation.IGNORED_SPURIOUS}
        if relations_used - allowed:
            raise ValidationError(
                "fate=update_existing permits only relations in "
                "{strengthens_target, ignored_spurious}"
            )
        if not merged_content:
            raise ValidationError("merged_content required for fate=update_existing")
