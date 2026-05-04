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


def test_commit_supersedes_rollback_on_tunnel_failure(palace, monkeypatch):
    """If tunnel creation fails mid-supersede, rollback restores old drawer + deletes new."""
    from mempalace import mcp_server

    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO seed\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    seed_id = seed["drawer_id"]

    pr = mcp_server.tool_prepare_write(
        content="DO seed\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody updated",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed")

    # Patch tool_create_tunnel to raise
    def boom(**kwargs):
        raise RuntimeError("simulated tunnel failure")
    monkeypatch.setattr(mcp_server, "tool_create_tunnel", boom)

    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="create_new",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "supersedes"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
    )
    assert "error" in cr
    audit = cr["audit_trail"]
    assert audit["rollback_attempted"] is True
    assert audit["rollback_completed"] is True

    # Old drawer body restored (status: confirmed)
    seed_after = mcp_server.tool_get_drawer(seed_id)
    assert "status: confirmed" in seed_after["content"]
    assert "status: superseded" not in seed_after["content"]


def test_commit_supersedes_rollback_on_tunnel_error_dict(palace, monkeypatch):
    """MemPalace primitives often return error dicts instead of raising. A
    tunnel error dict must abort the relation before status is flipped and
    trigger rollback of the newly-created drawer."""
    from mempalace import mcp_server

    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO seed dict\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody",
    )
    seed_id = seed["drawer_id"]

    pr = mcp_server.tool_prepare_write(
        content="DO seed dict\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody updated",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed")

    def returns_error_dict(**kwargs):
        return {"success": False, "error": "simulated dict-return tunnel failure"}
    monkeypatch.setattr(mcp_server, "tool_create_tunnel", returns_error_dict)

    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="create_new",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "supersedes"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
    )
    assert "error" in cr
    assert "dict-return tunnel failure" in cr["error"]
    audit = cr["audit_trail"]
    assert audit["rollback_attempted"] is True
    assert audit["rollback_completed"] is True
    assert len(audit["ops_attempted"]) != len(audit["ops_completed"])

    # Status was never flipped after the failed tunnel op.
    seed_after = mcp_server.tool_get_drawer(seed_id)
    assert "status: confirmed" in seed_after["content"]
    assert "status: superseded" not in seed_after["content"]


def test_commit_strengthens_rollback_on_update_failure(palace, monkeypatch):
    """If update_drawer fails for strengthens, original body must be restored."""
    from mempalace import mcp_server

    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO original\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nORIGINAL_BODY",
    )
    seed_id = seed["drawer_id"]

    pr = mcp_server.tool_prepare_write(
        content="DO original\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nORIGINAL_BODY",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed")

    # Patch tool_update_drawer to raise
    real = mcp_server.tool_update_drawer
    call_count = [0]
    def boom(**kwargs):
        call_count[0] += 1
        # First call (the strengthens update) fails; later calls (rollback) succeed
        if call_count[0] == 1:
            raise RuntimeError("simulated update failure")
        return real(**kwargs)
    monkeypatch.setattr(mcp_server, "tool_update_drawer", boom)

    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="update_existing",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "strengthens_target"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
        merged_content="MERGED_NEVER_LANDS",
    )
    assert "error" in cr
    seed_after = mcp_server.tool_get_drawer(seed_id)
    assert "ORIGINAL_BODY" in seed_after["content"]
    assert "MERGED_NEVER_LANDS" not in seed_after["content"]


def test_commit_strengthens_rollback_on_update_error_dict(palace, monkeypatch):
    """An update_drawer error dict must be treated as a failed commit, not a
    successful strengthens operation with fewer ops_completed than ops_attempted."""
    from mempalace import mcp_server

    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO original dict\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nORIGINAL_BODY_DICT",
    )
    seed_id = seed["drawer_id"]

    pr = mcp_server.tool_prepare_write(
        content="DO original dict\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nORIGINAL_BODY_DICT",
        wing="testwing", room="do", type="evidence", date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed")

    real = mcp_server.tool_update_drawer
    call_count = [0]
    def returns_error_dict(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"success": False, "error": "simulated dict-return update failure"}
        return real(**kwargs)
    monkeypatch.setattr(mcp_server, "tool_update_drawer", returns_error_dict)

    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="update_existing",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "strengthens_target"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
        merged_content="MERGED_DICT_NEVER_LANDS",
    )
    assert "error" in cr
    assert "dict-return update failure" in cr["error"]
    audit = cr["audit_trail"]
    assert audit["rollback_attempted"] is True
    assert audit["rollback_completed"] is True

    seed_after = mcp_server.tool_get_drawer(seed_id)
    assert "ORIGINAL_BODY_DICT" in seed_after["content"]
    assert "MERGED_DICT_NEVER_LANDS" not in seed_after["content"]
