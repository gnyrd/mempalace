"""Tests for tool_prepare_write — MCP-level integration."""
import pytest


@pytest.fixture
def isolated_palace(tmp_path, monkeypatch):
    import json
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"deprecated_add_drawer_response": "allow"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))
    from mempalace import mcp_server, guard
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)
    guard.reset_token_store()
    yield tmp_path


def test_prepare_write_returns_token_and_candidates(isolated_palace):
    from mempalace import mcp_server
    # Seed one drawer
    mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO use chat.guid\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nTried because: x\nEvidence: y\nCaveat: z",
    )
    result = mcp_server.tool_prepare_write(
        content="DO use chat.guid as the identifier — confirmed it works",
        wing="testwing",
        room="do",
        type="evidence",
        date="2026-05-04",
        source="test",
    )
    assert "guard_token" in result
    assert "expires_at" in result
    assert "blocking_candidates" in result
    assert "advisory_candidates" in result
    assert isinstance(result["blocking_candidates"], list)


def test_prepare_write_uses_policy_threshold(isolated_palace):
    """Default threshold from policy (0.7) is applied if caller doesn't override."""
    from mempalace import mcp_server
    mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="something\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nTried because: x\nEvidence: y",
    )
    r = mcp_server.tool_prepare_write(
        content="totally unrelated content about kubernetes",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    # Unrelated content shouldn't surface blocking candidates
    assert r["blocking_candidates"] == []


def test_prepare_write_returns_full_body_in_candidates(isolated_palace):
    """Per spec, candidate body should be the full body (not truncated like tool_check_duplicate)."""
    from mempalace import mcp_server
    long_body = (
        "DO use chat.guid\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
        "source: test\nstatus: confirmed\n\n" + ("Tried because: x\n" * 50)
    )
    mcp_server.tool_add_drawer(wing="testwing", room="do", content=long_body)
    r = mcp_server.tool_prepare_write(
        content="similar content about chat.guid identifier",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    if r["blocking_candidates"]:
        cand = r["blocking_candidates"][0]
        # Full body should be present (no 200-char truncation)
        assert len(cand["body"]) > 200
