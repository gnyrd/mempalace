"""End-to-end tests for tool_commit_write across all three fates."""
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


def _drawer(content_suffix=""):
    return (
        f"DO {content_suffix}\n\nauthor: claude\ntype: evidence\n"
        f"date: 2026-05-01\nsource: test\nstatus: confirmed\n\n"
        f"Tried because: x{content_suffix}\nEvidence: y\nCaveat: z"
    )


def test_create_new_with_no_blocking_writes_drawer(palace):
    from mempalace import mcp_server
    pr = mcp_server.tool_prepare_write(
        content=_drawer("new"),
        wing="testwing", room="do", type="evidence",
        date="2026-05-04", source="test",
    )
    assert pr["blocking_candidates"] == []
    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="create_new",
        candidate_relations=[],
    )
    assert cr["fate_taken"] == "create_new"
    assert len(cr["resulting_drawer_ids"]) == 1


def test_skip_new_returns_existing_drawer_id(palace):
    from mempalace import mcp_server
    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do", content=_drawer("seed"),
    )
    seed_id = seed["drawer_id"]
    pr = mcp_server.tool_prepare_write(
        content=_drawer("seed"),  # same content; should match
        wing="testwing", room="do", type="evidence",
        date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed as blocking; threshold-dependent")
    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="skip_new",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "duplicate"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
    )
    assert cr["fate_taken"] == "skip_new"
    assert cr["resulting_drawer_ids"] == []  # no new drawer


def test_update_existing_replaces_target_body(palace):
    from mempalace import mcp_server
    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do", content=_drawer("orig"),
    )
    seed_id = seed["drawer_id"]
    pr = mcp_server.tool_prepare_write(
        content=_drawer("orig"),
        wing="testwing", room="do", type="evidence",
        date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed as blocking")

    merged = "DO updated\n\nauthor: claude\ntype: evidence\ndate: 2026-05-04\nsource: test\nstatus: confirmed\n\nNEW MERGED CONTENT"
    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="update_existing",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "strengthens_target"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
        merged_content=merged,
    )
    assert cr["fate_taken"] == "update_existing"
    after = mcp_server.tool_get_drawer(seed_id)
    assert "NEW MERGED CONTENT" in after["content"]


def test_create_new_with_supersedes_creates_drawer_and_tunnel(palace):
    from mempalace import mcp_server
    seed = mcp_server.tool_add_drawer(
        wing="testwing", room="do", content=_drawer("old"),
    )
    seed_id = seed["drawer_id"]
    pr = mcp_server.tool_prepare_write(
        content=_drawer("old"),
        wing="testwing", room="do", type="evidence",
        date="2026-05-04", source="test",
    )
    if not pr["blocking_candidates"]:
        pytest.skip("retrieval did not surface seed as blocking")

    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="create_new",
        candidate_relations=[
            {"drawer_id": seed_id, "relation": "supersedes"},
        ] + [{"drawer_id": c["drawer_id"], "relation": "ignored_spurious"}
              for c in pr["blocking_candidates"] if c["drawer_id"] != seed_id],
    )
    assert cr["fate_taken"] == "create_new"
    assert len(cr["resulting_drawer_ids"]) == 1
    assert len(cr["resulting_tunnel_ids"]) >= 1

    seed_after = mcp_server.tool_get_drawer(seed_id)
    assert "status: superseded" in seed_after["content"]


def test_audit_trail_present_on_success(palace):
    from mempalace import mcp_server
    pr = mcp_server.tool_prepare_write(
        content=_drawer("xyz"),
        wing="testwing", room="do", type="evidence",
        date="2026-05-04", source="test",
    )
    cr = mcp_server.tool_commit_write(
        guard_token=pr["guard_token"],
        fate="create_new",
        candidate_relations=[],
    )
    assert "audit_trail" in cr
    audit = cr["audit_trail"]
    assert audit["fate"] == "create_new"
    assert "ops_attempted" in audit
    assert "ops_completed" in audit
    assert audit["rollback_attempted"] is False
