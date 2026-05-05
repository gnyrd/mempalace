"""Tests for tool_add_drawer policy gating per deprecated_add_drawer_response."""
import json
import pytest


@pytest.fixture
def palace_with_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(tmp_path / "policy.json"))
    from mempalace import mcp_server, guard
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)
    guard.reset_token_store()

    def _set_policy(response):
        (tmp_path / "policy.json").write_text(json.dumps({
            "deprecated_add_drawer_response": response,
        }))

    yield _set_policy


def test_allow_policy_lets_add_drawer_through(palace_with_policy):
    palace_with_policy("allow")
    from mempalace import mcp_server
    res = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    assert res.get("success") is True


def test_warn_policy_succeeds_with_warning(palace_with_policy, caplog):
    palace_with_policy("warn")
    from mempalace import mcp_server
    res = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    assert res.get("success") is True
    # Warning was logged
    assert any("deprecated" in rec.message.lower() or "use prepare_write" in rec.message.lower()
               for rec in caplog.records)


def test_refuse_policy_blocks_add_drawer(palace_with_policy):
    palace_with_policy("refuse")
    from mempalace import mcp_server
    res = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    assert res.get("success") is False
    assert "prepare_write" in res.get("error", "").lower()
