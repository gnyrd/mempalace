"""Tests for the guard token state (issuance, retrieval, expiry)."""
import time

import pytest

from mempalace.guard import TokenStore, TokenError


def test_issue_returns_unique_tokens():
    store = TokenStore(ttl_seconds=300)
    t1 = store.issue(content_hash="abc", blocking_ids=["x"], advisory_ids=[],
                    wing="w", room="do", type="evidence", prepared_content="hello",
                    effective_threshold=0.7, effective_advisory_floor=0.5)
    t2 = store.issue(content_hash="abc", blocking_ids=["x"], advisory_ids=[],
                    wing="w", room="do", type="evidence", prepared_content="hello",
                    effective_threshold=0.7, effective_advisory_floor=0.5)
    assert t1 != t2


def test_get_returns_record():
    store = TokenStore(ttl_seconds=300)
    tok = store.issue(content_hash="abc", blocking_ids=["x", "y"], advisory_ids=["z"],
                      wing="w", room="do", type="evidence", prepared_content="hello",
                      effective_threshold=0.7, effective_advisory_floor=0.5)
    rec = store.get(tok)
    assert rec.content_hash == "abc"
    assert rec.blocking_ids == ["x", "y"]
    assert rec.advisory_ids == ["z"]
    assert rec.wing == "w"
    assert rec.prepared_content == "hello"


def test_get_unknown_token_raises():
    store = TokenStore(ttl_seconds=300)
    with pytest.raises(TokenError, match="unknown"):
        store.get("not-a-token")


def test_expired_token_raises_and_is_purged():
    store = TokenStore(ttl_seconds=0)  # immediate expiry
    tok = store.issue(content_hash="abc", blocking_ids=[], advisory_ids=[],
                      wing="w", room="do", type="evidence", prepared_content="hello",
                      effective_threshold=0.7, effective_advisory_floor=0.5)
    time.sleep(0.01)
    with pytest.raises(TokenError, match="expired"):
        store.get(tok)
    # And the expired token is no longer retrievable even with a fresh check
    with pytest.raises(TokenError):
        store.get(tok)


def test_consume_removes_token():
    store = TokenStore(ttl_seconds=300)
    tok = store.issue(content_hash="abc", blocking_ids=[], advisory_ids=[],
                      wing="w", room="do", type="evidence", prepared_content="hello",
                      effective_threshold=0.7, effective_advisory_floor=0.5)
    rec = store.consume(tok)
    assert rec.content_hash == "abc"
    with pytest.raises(TokenError, match="unknown"):
        store.get(tok)


def test_purge_expired_removes_old_tokens():
    store = TokenStore(ttl_seconds=0)
    for _ in range(5):
        store.issue(content_hash="x", blocking_ids=[], advisory_ids=[],
                    wing="w", room="do", type="evidence", prepared_content="x",
                    effective_threshold=0.7, effective_advisory_floor=0.5)
    time.sleep(0.01)
    store.purge_expired()
    assert store.size() == 0
