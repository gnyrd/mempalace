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
from dataclasses import dataclass, field
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
