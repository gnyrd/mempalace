"""Write-guard core.

Owns the prepare/commit state machine: token issuance, fate-vs-relation
validation, per-relation operations (calling existing tool_create_tunnel /
tool_update_drawer), preimage capture, compensating rollback, audit-trail
builder.

This file is intentionally a single module to keep related state machinery
co-located. Sub-sections (tokens, validation, operations, atomicity) are
clearly separated for readability.
"""
import re
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


# ============================================================================
# Process-wide singletons (server-side)
# ============================================================================


_token_store: Optional[TokenStore] = None


def get_token_store() -> TokenStore:
    """Return the process-wide TokenStore, creating on first use.

    Reads policy at first access; TTL set from policy.token_ttl_seconds.
    """
    global _token_store
    if _token_store is None:
        from mempalace.policy import load_policy
        pol = load_policy()
        _token_store = TokenStore(ttl_seconds=pol.token_ttl_seconds)
    return _token_store


def reset_token_store() -> None:
    """Test helper: drop the singleton so the next get_token_store() rebuilds with current policy."""
    global _token_store
    _token_store = None

# ============================================================================
# Per-relation operations
#
# Each function returns an op_log: list of {"op": str, "args": dict, "result": ...}
# entries documenting what was attempted and what happened. The caller (commit_write)
# integrates these into the final audit_trail.
#
# Tunnel directions (per spec v1.2):
#   - supersedes: target → new (old points to its replacement)
#   - resolves_contradiction: contradiction → resolution
#   - related_distinct: per association.direction
# ============================================================================


class GuardOpError(RuntimeError):
    """Raised when a per-relation operation returns an error-shaped result.

    MemPalace primitives often return {"error": "..."} or
    {"success": False, "error": "..."} instead of raising. Treat those exactly
    like exceptions so commit_write can roll back partial state.
    """

    def __init__(self, op: dict, op_log: list = None):
        self.op = op
        self.op_log = list(op_log or [op])
        result = op.get("result") or {}
        msg = result.get("error") or f"operation returned unsuccessful result: {result}"
        super().__init__(f"{op.get('op', 'operation')} failed: {msg}")


def op_failed(op: dict) -> bool:
    """Return True if an op-log entry represents a failed primitive call."""
    result = op.get("result") or {}
    return bool(result.get("error") or result.get("success") is False)


def raise_if_op_failed(op: dict, op_log: list = None) -> None:
    """Raise GuardOpError for error-dict primitive results."""
    if op_failed(op):
        raise GuardOpError(op=op, op_log=op_log)


def _set_status_in_body(body: str, new_status: str) -> str:
    """Replace the typed-memory `status:` line in a drawer body."""
    return re.sub(
        r"^status:\s*\S+",
        f"status: {new_status}",
        body,
        count=1,
        flags=re.MULTILINE,
    )


def _drawer_get(drawer_id: str) -> dict:
    from mempalace import mcp_server
    return mcp_server.tool_get_drawer(drawer_id)


def _tag_superseded(drawer_id: str) -> dict:
    """Read drawer; flip status: -> superseded; write back via tool_update_drawer."""
    from mempalace import mcp_server
    cur = _drawer_get(drawer_id)
    if not cur.get("content"):
        return {"op": "tag_superseded", "args": {"drawer_id": drawer_id},
                "result": {"error": "drawer missing"}}
    new_body = _set_status_in_body(cur["content"], "superseded")
    res = mcp_server.tool_update_drawer(drawer_id=drawer_id, content=new_body)
    return {"op": "tag_superseded", "args": {"drawer_id": drawer_id}, "result": res}


def _create_tunnel(source_drawer_id: str, target_drawer_id: str, kind: str = None) -> dict:
    """Wrap tool_create_tunnel. We must look up wing/room for both drawers."""
    from mempalace import mcp_server
    src = _drawer_get(source_drawer_id)
    tgt = _drawer_get(target_drawer_id)
    args = {
        "source_wing": src.get("wing"),
        "source_room": src.get("room"),
        "source_drawer_id": source_drawer_id,
        "target_wing": tgt.get("wing"),
        "target_room": tgt.get("room"),
        "target_drawer_id": target_drawer_id,
    }
    if kind is not None:
        args["label"] = kind
    res = mcp_server.tool_create_tunnel(**args)
    return {"op": "create_tunnel", "args": args, "result": res}


def apply_relation(
    *,
    relation: str,
    target_drawer_id: Optional[str],
    new_drawer_id: Optional[str],
    association: Optional[dict],
    merged_content: Optional[str],
) -> list:
    """Execute the operations for one candidate relation.

    Returns a list of op-log entries. Caller appends these to the audit trail.

    The new drawer (when one is being created) MUST already exist before this is
    called; the create_new fate is the caller's responsibility (it adds the new
    drawer once before iterating relations).
    """
    op_log = []

    def _append_checked(op: dict) -> None:
        op_log.append(op)
        raise_if_op_failed(op, op_log=op_log)

    if relation == Relation.IGNORED_SPURIOUS:
        return op_log

    if relation == Relation.DUPLICATE:
        # Caller chose fate=skip_new; nothing per-relation
        return op_log

    if relation == Relation.SUPERSEDES:
        assert new_drawer_id, "supersedes requires a new drawer"
        # target → new (old points to its replacement)
        _append_checked(_create_tunnel(target_drawer_id, new_drawer_id, kind="supersedes"))
        _append_checked(_tag_superseded(target_drawer_id))
        return op_log

    if relation == Relation.RESOLVES_CONTRADICTION:
        assert new_drawer_id, "resolves_contradiction requires a new drawer"
        # contradiction → resolution
        _append_checked(_create_tunnel(target_drawer_id, new_drawer_id, kind="resolves"))
        _append_checked(_tag_superseded(target_drawer_id))
        return op_log

    if relation == Relation.RELATED_DISTINCT:
        assert new_drawer_id, "related_distinct requires a new drawer"
        assert association, "related_distinct requires association"
        direction = association.get("direction", "from_new")
        if direction == "from_new":
            _append_checked(_create_tunnel(new_drawer_id, target_drawer_id, kind="related"))
        elif direction == "to_new":
            _append_checked(_create_tunnel(target_drawer_id, new_drawer_id, kind="related"))
        else:
            raise ValueError(f"unknown association direction: {direction!r}")
        return op_log

    if relation == Relation.STRENGTHENS_TARGET:
        assert merged_content, "strengthens_target requires merged_content"
        from mempalace import mcp_server
        res = mcp_server.tool_update_drawer(drawer_id=target_drawer_id, content=merged_content)
        _append_checked({"op": "update_drawer", "args": {"drawer_id": target_drawer_id}, "result": res})
        return op_log

    raise ValueError(f"unhandled relation: {relation!r}")
