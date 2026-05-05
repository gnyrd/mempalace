"""Storage-side wing normalization regression tests.

The miner already canonicalizes wing names (`-` -> `_`, lowercased) when
populating wings from project dirnames, so all queries assume the canonical
form. The MCP write boundary previously did not normalize, so a caller
passing `wing="vault-ops"` would land a drawer under the literal hyphenated
wing while existing content lived under `vault_ops`. This split knowledge
across two wings invisibly. The 2026-05-05 cleanup migrated 5 such drawers;
these tests pin the normalization at the write boundary so it can't recur.
"""
import json
import pytest


@pytest.fixture
def palace_allow(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path / "palace"))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(tmp_path / "policy.json"))
    (tmp_path / "policy.json").write_text(json.dumps({
        "deprecated_add_drawer_response": "allow",
    }))
    from mempalace import mcp_server, guard
    monkeypatch.setattr(mcp_server, "_client", None, raising=False)
    monkeypatch.setattr(mcp_server, "_collection", None, raising=False)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None, raising=False)
    guard.reset_token_store()
    return mcp_server


_BODY = (
    "author: claude\ntype: evidence\ndate: 2026-05-05\n"
    "source: test_wing_normalization\nstatus: confirmed\n\n"
    "Tried because: pin the wing normalization\n"
    "Evidence: passes mempalace test suite\n"
    "Caveat: applies only at MCP write boundary"
)


def test_add_drawer_normalizes_hyphenated_wing(palace_allow):
    """A hyphenated wing slug must land the drawer under the canonical
    underscore form, with the canonical wing in metadata and the canonical
    wing prefix in the deterministic drawer_id."""
    res = palace_allow.tool_add_drawer(
        wing="vault-ops", room="do",
        content=f"DO normalize wings at the MCP write boundary\n\n{_BODY}",
    )
    assert res["success"] is True
    assert res["wing"] == "vault_ops"
    assert res["drawer_id"].startswith("drawer_vault_ops_do_")

    listed = palace_allow.tool_list_drawers(wing="vault_ops")
    assert listed["count"] == 1
    assert "vault-ops" not in {d["wing"] for d in listed["drawers"]}


def test_add_drawer_normalizes_uppercase_and_spaces(palace_allow):
    """normalize_wing_name lowercases AND collapses spaces to underscores —
    not just hyphens."""
    res = palace_allow.tool_add_drawer(
        wing="My Project", room="do",
        content=f"DO normalize uppercase + space wings\n\n{_BODY}",
    )
    assert res["success"] is True
    assert res["wing"] == "my_project"


def test_prepare_write_normalizes_wing(palace_allow):
    """prepare_write must canonicalize the wing so retrieval queries the
    right wing AND the rec.wing baked into the guard token is canonical
    (so commit_write's call to _tool_add_drawer_unguarded passes the
    canonical wing all the way through)."""
    # Seed an existing drawer under the canonical wing.
    seed = palace_allow.tool_add_drawer(
        wing="vault_ops", room="do",
        content=f"DO seed for prepare_write retrieval test\n\n{_BODY}",
    )
    assert seed["success"] is True

    # Call prepare_write with the hyphenated form. Retrieval should still
    # see the seeded drawer because both sides normalize.
    prep = palace_allow.tool_prepare_write(
        content=f"DO seed for prepare_write retrieval test\n\n{_BODY}",
        wing="vault-ops", room="do",
        type="evidence", date="2026-05-05",
        source="test_wing_normalization",
    )
    assert "guard_token" in prep
    blocking_wings = {c["wing"] for c in prep["blocking_candidates"]}
    # The seeded drawer is the obvious blocker — its wing in the candidate
    # serialization must be the canonical form.
    assert "vault_ops" in blocking_wings
    assert "vault-ops" not in blocking_wings


def test_create_tunnel_normalizes_endpoint_wings(palace_allow):
    """tool_create_tunnel must store endpoint wings in canonical form so
    drawer-aware tunnel identity hashing (which already normalizes via
    palace_graph._normalize_wing) and downstream listing/follow operations
    agree on the wing string."""
    a = palace_allow.tool_add_drawer(
        wing="vault_ops", room="do",
        content=f"DO source endpoint\n\n{_BODY}",
    )
    b = palace_allow.tool_add_drawer(
        wing="mempalace", room="do",
        content=f"DO target endpoint\n\n{_BODY}",
    )
    res = palace_allow.tool_create_tunnel(
        source_wing="vault-ops", source_room="do",
        target_wing="MemPalace", target_room="do",
        source_drawer_id=a["drawer_id"],
        target_drawer_id=b["drawer_id"],
        label="related",
    )
    assert "id" in res
    assert res["source"]["wing"] == "vault_ops"
    assert res["target"]["wing"] == "mempalace"
