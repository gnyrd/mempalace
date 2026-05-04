"""Tests for per-relation operations (the per-candidate state changes)."""
import json
import pytest


@pytest.fixture
def palace_with_drawers(tmp_path, monkeypatch):
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"deprecated_add_drawer_response": "allow"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))
    from mempalace import mcp_server, guard
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)
    guard.reset_token_store()

    # Seed two drawers
    a = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO original\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody A",
    )
    b = mcp_server.tool_add_drawer(
        wing="testwing", room="dont",
        content="DONT something\n\nauthor: claude\ntype: evidence\ndate: 2026-05-01\n"
                "source: test\nstatus: confirmed\n\nbody B",
    )
    yield tmp_path, a["drawer_id"], b["drawer_id"]


def test_apply_relation_supersedes_creates_old_to_new_tunnel(palace_with_drawers):
    """relation=supersedes: tag old drawer status: superseded; tunnel target → new (per spec v1.2)."""
    from mempalace import mcp_server, guard

    tmp, a_id, _ = palace_with_drawers
    new = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO new replacement\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nbody NEW",
    )
    new_id = new["drawer_id"]

    op_log = guard.apply_relation(
        relation=guard.Relation.SUPERSEDES,
        target_drawer_id=a_id,
        new_drawer_id=new_id,
        association=None,
        merged_content=None,
    )

    # Tunnel direction: target → new (old points to its replacement, per v1.2 conventions).
    # NOTE (Task 7 plan deviation): tool_list_tunnels returns a list directly
    # (or {"error": ...} on bad arg), not {"tunnels": [...]}. Tunnel structure is
    # nested: {source: {wing, room, drawer_id}, target: {...}}. Plan v2.0 used
    # a flat schema that doesn't match MemPalace v3.3.3. Flagged back to Codex
    # for v2.2 plan patch.
    tunnels = mcp_server.tool_list_tunnels()
    assert any(
        t["source"]["room"] == "do"
        and t["source"].get("drawer_id") == a_id
        and t["target"].get("drawer_id") == new_id
        for t in tunnels
    ), f"expected tunnel a_id→new_id; saw {tunnels}"

    # Old drawer body now has status: superseded
    a_after = mcp_server.tool_get_drawer(a_id)
    assert "status: superseded" in a_after["content"]

    # Op log records what happened
    assert any(o["op"] == "tag_superseded" for o in op_log)
    assert any(o["op"] == "create_tunnel" for o in op_log)


def test_apply_relation_resolves_contradiction(palace_with_drawers):
    """relation=resolves_contradiction: tag old; tunnel contradiction → resolution."""
    from mempalace import mcp_server, guard

    tmp, a_id, _ = palace_with_drawers
    res = mcp_server.tool_add_drawer(
        wing="testwing", room="general",
        content="RESOLUTION x\n\nauthor: claude\ntype: resolution\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nmerged stance",
    )
    res_id = res["drawer_id"]

    op_log = guard.apply_relation(
        relation=guard.Relation.RESOLVES_CONTRADICTION,
        target_drawer_id=a_id,  # treating a_id as the contradiction for the test
        new_drawer_id=res_id,
        association=None,
        merged_content=None,
    )

    a_after = mcp_server.tool_get_drawer(a_id)
    assert "status: superseded" in a_after["content"]
    # Schema: tool_list_tunnels returns a list; tunnels are nested
    # {source: {...}, target: {...}}. (Task 7 plan deviation — see supersedes test.)
    tunnels = mcp_server.tool_list_tunnels()
    assert any(
        t["source"].get("drawer_id") == a_id and t["target"].get("drawer_id") == res_id
        for t in tunnels
    )
    assert any(o["op"] == "tag_superseded" for o in op_log)


def test_apply_relation_related_distinct_from_new(palace_with_drawers):
    from mempalace import mcp_server, guard

    tmp, a_id, _ = palace_with_drawers
    new = mcp_server.tool_add_drawer(
        wing="testwing", room="do",
        content="DO new\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\n"
                "source: test\nstatus: confirmed\n\nnew body",
    )
    new_id = new["drawer_id"]

    op_log = guard.apply_relation(
        relation=guard.Relation.RELATED_DISTINCT,
        target_drawer_id=a_id,
        new_drawer_id=new_id,
        association={"kind": "tunnel", "direction": "from_new"},
        merged_content=None,
    )
    # Schema: see supersedes test.
    tunnels = mcp_server.tool_list_tunnels()
    # direction=from_new means new_id → a_id
    assert any(
        t["source"].get("drawer_id") == new_id and t["target"].get("drawer_id") == a_id
        for t in tunnels
    )
    assert any(o["op"] == "create_tunnel" for o in op_log)


def test_apply_relation_strengthens_target_updates_body(palace_with_drawers):
    from mempalace import mcp_server, guard

    tmp, a_id, _ = palace_with_drawers

    op_log = guard.apply_relation(
        relation=guard.Relation.STRENGTHENS_TARGET,
        target_drawer_id=a_id,
        new_drawer_id=None,
        association=None,
        merged_content="NEW MERGED BODY",
    )
    a_after = mcp_server.tool_get_drawer(a_id)
    assert "NEW MERGED BODY" in a_after["content"]
    assert any(o["op"] == "update_drawer" for o in op_log)


def test_apply_relation_ignored_spurious_is_noop(palace_with_drawers):
    from mempalace import mcp_server, guard

    tmp, a_id, _ = palace_with_drawers
    a_before = mcp_server.tool_get_drawer(a_id)
    op_log = guard.apply_relation(
        relation=guard.Relation.IGNORED_SPURIOUS,
        target_drawer_id=a_id,
        new_drawer_id=None,
        association=None,
        merged_content=None,
    )
    a_after = mcp_server.tool_get_drawer(a_id)
    assert a_before["content"] == a_after["content"]
    assert op_log == []  # no operations executed
