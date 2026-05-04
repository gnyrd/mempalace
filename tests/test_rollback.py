"""Tests for preimage capture + compensating rollback in commit_write.

Strategy: build operation contexts manually, inject failures, verify state
after rollback matches state before the operation began.
"""
import json
import pytest


@pytest.fixture
def palace(tmp_path, monkeypatch):
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


def test_preimage_capture_records_body_and_status(palace):
    from mempalace import mcp_server, guard
    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO test\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    ctx = guard.OperationContext()
    ctx.capture_preimage(a["drawer_id"])
    pre = ctx.preimages[a["drawer_id"]]
    assert "status: confirmed" in pre["content"]
    assert pre["wing"] == "testwing"
    assert pre["room"] == "do"


def test_rollback_restores_body_after_status_flip(palace):
    from mempalace import mcp_server, guard
    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO test\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    a_id = a["drawer_id"]
    ctx = guard.OperationContext()
    ctx.capture_preimage(a_id)

    # Flip status; record op
    ctx.record_op(guard._tag_superseded(a_id))
    a_mid = mcp_server.tool_get_drawer(a_id)
    assert "status: superseded" in a_mid["content"]

    # Rollback
    ctx.rollback()
    a_after = mcp_server.tool_get_drawer(a_id)
    assert "status: confirmed" in a_after["content"]
    assert "status: superseded" not in a_after["content"]


def test_rollback_deletes_added_drawer(palace):
    from mempalace import mcp_server, guard
    ctx = guard.OperationContext()
    new = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO new\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    new_id = new["drawer_id"]
    ctx.record_added_drawer(new_id)
    ctx.rollback()
    after = mcp_server.tool_get_drawer(new_id)
    assert after.get("error") or after.get("found") is False, f"expected drawer deleted; got {after}"


def test_rollback_removes_created_tunnels(palace):
    from mempalace import mcp_server, guard
    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO a\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    b = mcp_server.tool_add_drawer(
        wing="testwing", room="dont",
        content="DONT b\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    ctx = guard.OperationContext()
    op = guard._create_tunnel(a["drawer_id"], b["drawer_id"])
    ctx.record_op(op)
    # NOTE (Task 8 plan deviation): tool_create_tunnel returns the stored
    # tunnel dict whose key is "id" (not "tunnel_id"). Plan v2.0 used
    # "tunnel_id" everywhere — patched here. tool_list_tunnels returns a
    # list directly, not {"tunnels": [...]}. Flagged for Codex v2.2.
    tunnel_id = op["result"].get("id")
    assert tunnel_id, f"expected id in {op}"

    ctx.rollback()

    tunnels = mcp_server.tool_list_tunnels()
    assert not any(t.get("id") == tunnel_id for t in tunnels)


def test_rollback_handles_failure_gracefully(palace):
    """If rollback itself fails, ctx records rollback_completed=False AND
    captures the per-op error in rollback_errors so audit trail can surface it."""
    from mempalace import mcp_server, guard
    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nb",
    )
    ctx = guard.OperationContext()
    ctx.capture_preimage(a["drawer_id"])
    # Manually corrupt preimage to trigger rollback failure
    ctx.preimages[a["drawer_id"]]["content"] = None

    ctx.rollback()
    assert ctx.rollback_attempted is True
    assert ctx.rollback_completed is False
    assert len(ctx.rollback_errors) >= 1
    err = ctx.rollback_errors[0]
    assert err["op"] == "restore_preimage"
    assert err["id"] == a["drawer_id"]
    assert "corrupt preimage" in err["error"]


def test_rollback_swallowed_tunnel_error_recorded(palace, monkeypatch):
    """A tunnel-delete failure must appear in rollback_errors and prevent
    rollback_completed=True (catches the bug Codex flagged in v1).

    Note (Task 8 plan deviation): test relies on record_op using the actual
    tunnel-id key 'id' (not the plan's 'tunnel_id'). See guard.py patch
    in this commit. Flagged for Codex v2.2.
    """
    from mempalace import mcp_server, guard

    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO a\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    b = mcp_server.tool_add_drawer(
        wing="testwing", room="dont",
        content="DONT b\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    ctx = guard.OperationContext()
    op = guard._create_tunnel(a["drawer_id"], b["drawer_id"])
    ctx.record_op(op)

    # Patch tool_delete_tunnel to fail
    def fail(**kwargs):
        raise RuntimeError("tunnel delete blew up")
    monkeypatch.setattr(mcp_server, "tool_delete_tunnel", fail)

    ctx.rollback()
    assert ctx.rollback_completed is False
    assert any("tunnel" in e["op"] and "blew up" in e["error"]
               for e in ctx.rollback_errors)
