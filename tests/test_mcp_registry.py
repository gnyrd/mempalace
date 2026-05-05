"""MCP TOOLS registry smoke tests.

Codex Checkpoint B P1 #1 caught that `tool_prepare_write` and
`tool_commit_write` were defined as Python functions but never registered
in the `TOOLS` dict, so MCP clients couldn't see or call them. This file
guards against that regression — if a future change defines a new
write-guard handler but forgets the registry entry, these tests fail.
"""
from mempalace import mcp_server


def test_prepare_write_registered():
    assert "mempalace_prepare_write" in mcp_server.TOOLS
    entry = mcp_server.TOOLS["mempalace_prepare_write"]
    assert entry["handler"] is mcp_server.tool_prepare_write
    schema = entry["input_schema"]
    assert schema["type"] == "object"
    # Required args per the prepare/commit two-step protocol
    for required in ("content", "wing", "room", "type", "date", "source"):
        assert required in schema["properties"], (
            f"prepare_write schema missing property {required!r}"
        )
    assert set(schema["required"]) == {"content", "wing", "room", "type", "date", "source"}


def test_commit_write_registered():
    assert "mempalace_commit_write" in mcp_server.TOOLS
    entry = mcp_server.TOOLS["mempalace_commit_write"]
    assert entry["handler"] is mcp_server.tool_commit_write
    schema = entry["input_schema"]
    assert schema["type"] == "object"
    for required in ("guard_token", "fate", "candidate_relations",
                     "advisory_acknowledged", "merged_content"):
        assert required in schema["properties"], (
            f"commit_write schema missing property {required!r}"
        )
    # Only guard_token + fate are mandatory at the schema level; the rest
    # are validated by validate_commit_args inside the handler.
    assert set(schema["required"]) == {"guard_token", "fate"}


def test_prepare_and_commit_appear_in_registry_order():
    """Sanity: the prepare/commit pair lives between check_duplicate and
    add_drawer, matching the file's physical layout. Catches accidental
    reordering or duplication."""
    keys = list(mcp_server.TOOLS.keys())
    cd_idx = keys.index("mempalace_check_duplicate")
    pw_idx = keys.index("mempalace_prepare_write")
    cw_idx = keys.index("mempalace_commit_write")
    ad_idx = keys.index("mempalace_add_drawer")
    assert cd_idx < pw_idx < cw_idx < ad_idx, (
        f"unexpected TOOLS ordering: check_duplicate={cd_idx}, "
        f"prepare_write={pw_idx}, commit_write={cw_idx}, add_drawer={ad_idx}"
    )
