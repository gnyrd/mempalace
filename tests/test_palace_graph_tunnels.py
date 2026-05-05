"""Tests for explicit tunnel helpers in mempalace.palace_graph."""

import hashlib
import os
import stat
import sys
from unittest.mock import MagicMock, patch

import pytest

with patch.dict("sys.modules", {"chromadb": MagicMock()}):
    import mempalace.palace_graph as palace_graph


def _use_tmp_tunnel_file(monkeypatch, tmp_path):
    tunnel_file = tmp_path / "tunnels.json"
    monkeypatch.setattr(palace_graph, "_TUNNEL_FILE", str(tunnel_file))
    return tunnel_file


class TestTunnelStorage:
    def test_load_tunnels_missing_file_returns_empty_list(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        assert palace_graph._load_tunnels() == []

    def test_load_tunnels_corrupt_file_returns_empty_list(self, tmp_path, monkeypatch):
        tunnel_file = _use_tmp_tunnel_file(monkeypatch, tmp_path)
        tunnel_file.write_text("{not valid json", encoding="utf-8")
        assert palace_graph._load_tunnels() == []

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        tunnels = [
            {
                "id": "abc123",
                "source": {"wing": "wing_code", "room": "auth"},
                "target": {"wing": "wing_people", "room": "users"},
                "label": "same concept",
            }
        ]
        palace_graph._save_tunnels(tunnels)
        assert palace_graph._load_tunnels() == tunnels

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX file-permission bits only apply on Unix-like systems",
    )
    def test_save_tunnels_restricts_permissions(self, tmp_path, monkeypatch):
        """Regression for #1165 — tunnels.json reveals cross-wing links and
        must not be world-readable on shared Linux/multi-user systems."""
        tunnel_file = _use_tmp_tunnel_file(monkeypatch, tmp_path)
        palace_graph._save_tunnels(
            [
                {
                    "id": "x",
                    "source": {"wing": "a", "room": "r1"},
                    "target": {"wing": "b", "room": "r2"},
                    "label": "",
                }
            ]
        )

        file_mode = stat.S_IMODE(os.stat(tunnel_file).st_mode)
        assert file_mode == 0o600, f"tunnels.json mode is {oct(file_mode)}, expected 0o600"

        parent_mode = stat.S_IMODE(os.stat(tunnel_file.parent).st_mode)
        assert (
            parent_mode == 0o700
        ), f"tunnels.json parent dir mode is {oct(parent_mode)}, expected 0o700"


class TestExplicitTunnels:
    def test_create_tunnel_deduplicates_reverse_order_and_updates_label(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        first = palace_graph.create_tunnel(
            "wing_code", "auth", "wing_people", "users", label="same concept"
        )
        second = palace_graph.create_tunnel(
            "wing_people", "users", "wing_code", "auth", label="updated label"
        )

        assert first["id"] == second["id"]
        assert len(palace_graph.list_tunnels()) == 1
        assert second["label"] == "updated label"
        assert second["created_at"] == first["created_at"]
        assert "updated_at" in second

    def test_create_tunnel_rejects_empty_names(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        with pytest.raises(ValueError):
            palace_graph.create_tunnel("", "auth", "wing_people", "users")

    def test_list_tunnels_filters_by_either_side(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel("wing_code", "auth", "wing_people", "users", label="A")
        palace_graph.create_tunnel("wing_ops", "deploy", "wing_people", "users", label="B")

        assert len(palace_graph.list_tunnels()) == 2
        assert len(palace_graph.list_tunnels("wing_people")) == 2
        assert len(palace_graph.list_tunnels("wing_code")) == 1

    def test_delete_tunnel_removes_saved_tunnel(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        tunnel = palace_graph.create_tunnel(
            "wing_code", "auth", "wing_people", "users", label="same concept"
        )

        assert palace_graph.delete_tunnel(tunnel["id"]) == {"deleted": tunnel["id"]}
        assert palace_graph.list_tunnels() == []

    def test_follow_tunnels_returns_direction_and_preview(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )

        col = MagicMock()
        col.get.return_value = {
            "ids": ["drawer_users_1"],
            "documents": ["A" * 400],
            "metadatas": [{}],
        }

        outgoing = palace_graph.follow_tunnels("wing_code", "auth", col=col)
        assert len(outgoing) == 1
        assert outgoing[0]["direction"] == "outgoing"
        assert outgoing[0]["connected_wing"] == "wing_people"
        assert outgoing[0]["connected_room"] == "users"
        assert outgoing[0]["drawer_id"] == "drawer_users_1"
        assert len(outgoing[0]["drawer_preview"]) == 300

        incoming = palace_graph.follow_tunnels("wing_people", "users", col=col)
        assert len(incoming) == 1
        assert incoming[0]["direction"] == "incoming"
        assert incoming[0]["connected_wing"] == "wing_code"

    def test_follow_tunnels_returns_connections_even_if_collection_lookup_fails(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )

        col = MagicMock()
        col.get.side_effect = RuntimeError("boom")

        connections = palace_graph.follow_tunnels("wing_code", "auth", col=col)
        assert len(connections) == 1
        assert "drawer_preview" not in connections[0]


class TestTopicTunnels:
    """Cross-wing topic tunnels (issue #1180).

    When two wings share confirmed TOPIC labels above a configurable
    threshold, a symmetric tunnel is created between them. Tunnels are
    routed through the existing ``create_tunnel`` storage so they share
    dedup and persistence with explicit tunnels.
    """

    def test_compute_topic_tunnels_creates_link_for_shared_topic(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI"],
            "wing_beta": ["OpenAPI", "Kubernetes"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        assert len(created) == 1
        assert created[0]["source"]["wing"] in {"wing_alpha", "wing_beta"}
        assert created[0]["target"]["wing"] in {"wing_alpha", "wing_beta"}
        # Room is namespaced with the ``topic:`` prefix so it can't collide
        # with a literal folder-derived room of the same name. Casing of the
        # topic is preserved for display.
        assert created[0]["source"]["room"] == "topic:OpenAPI"
        assert created[0]["target"]["room"] == "topic:OpenAPI"
        assert created[0]["kind"] == "topic"
        # Label carries the human-readable topic without the prefix.
        assert "OpenAPI" in created[0]["label"]
        assert "topic:OpenAPI" not in created[0]["label"]

        # Tunnel is retrievable via the standard list_tunnels API.
        listed = palace_graph.list_tunnels()
        assert len(listed) == 1
        assert listed[0]["id"] == created[0]["id"]

    def test_compute_topic_tunnels_no_link_below_threshold(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI"],
            "wing_beta": ["OpenAPI", "Kubernetes"],
        }
        # min_count=2 requires two overlapping topics — only one shared.
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=2)
        assert created == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_above_threshold_creates_per_topic_links(
        self, tmp_path, monkeypatch
    ):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["Angular", "OpenAPI", "Postgres"],
            "wing_beta": ["Angular", "OpenAPI", "Redis"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=2)
        # Two shared topics × one wing pair = two tunnels.
        rooms = sorted(t["source"]["room"] for t in created)
        assert rooms == ["topic:Angular", "topic:OpenAPI"]

    def test_compute_topic_tunnels_case_insensitive_overlap(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["openapi"],
            "wing_beta": ["OpenAPI"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        assert len(created) == 1

    def test_compute_topic_tunnels_empty_input_is_noop(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        assert palace_graph.compute_topic_tunnels({}) == []
        assert palace_graph.compute_topic_tunnels({"wing_a": []}) == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_three_wings_pairwise(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_a": ["foo"],
            "wing_b": ["foo"],
            "wing_c": ["foo"],
        }
        created = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        # 3 wings sharing the same topic → C(3,2) = 3 pairs → 3 tunnels.
        assert len(created) == 3
        endpoint_pairs = {
            tuple(sorted([t["source"]["wing"], t["target"]["wing"]])) for t in created
        }
        assert endpoint_pairs == {
            ("wing_a", "wing_b"),
            ("wing_a", "wing_c"),
            ("wing_b", "wing_c"),
        }

    def test_topic_tunnels_for_wing_only_links_that_wing(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_a": ["foo", "bar"],
            "wing_b": ["foo"],
            "wing_c": ["bar"],
        }
        # wing_a should link to both b (via foo) and c (via bar).
        created = palace_graph.topic_tunnels_for_wing("wing_a", topics_by_wing)
        endpoint_pairs = {
            tuple(sorted([t["source"]["wing"], t["target"]["wing"]])) for t in created
        }
        assert endpoint_pairs == {("wing_a", "wing_b"), ("wing_a", "wing_c")}
        # The b-c pair is NOT created because wing_a's incremental pass
        # only computes pairs that include wing_a.
        assert len(palace_graph.list_tunnels()) == 2

    def test_topic_tunnels_for_wing_unknown_wing_is_noop(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {"wing_a": ["foo"], "wing_b": ["foo"]}
        assert palace_graph.topic_tunnels_for_wing("wing_missing", topics_by_wing) == []
        assert palace_graph.list_tunnels() == []

    def test_compute_topic_tunnels_dedupe_on_recompute(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        topics_by_wing = {
            "wing_alpha": ["OpenAPI"],
            "wing_beta": ["OpenAPI"],
        }
        first = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        second = palace_graph.compute_topic_tunnels(topics_by_wing, min_count=1)
        # create_tunnel is symmetric/dedupe — repeated computation should
        # not multiply the stored tunnels.
        assert first[0]["id"] == second[0]["id"]
        assert len(palace_graph.list_tunnels()) == 1

    def test_topic_tunnel_room_does_not_collide_with_literal_room(self, tmp_path, monkeypatch):
        """Regression: a literal "Angular" folder-room and a topic tunnel
        for "Angular" must resolve to distinct endpoints so ``follow_tunnels``
        from the real room doesn't accidentally surface topic connections
        (issue raised in review of #1184)."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        # Explicit tunnel anchored at a literal "Angular" room in wing_alpha.
        palace_graph.create_tunnel(
            "wing_alpha", "Angular", "wing_gamma", "frontend", label="explicit"
        )
        # Topic tunnel between the same wings that share the "Angular" topic.
        palace_graph.compute_topic_tunnels(
            {"wing_alpha": ["Angular"], "wing_beta": ["Angular"]}, min_count=1
        )

        # follow_tunnels on the literal Angular room only sees the explicit link.
        literal = palace_graph.follow_tunnels("wing_alpha", "Angular")
        assert len(literal) == 1
        assert literal[0]["connected_wing"] == "wing_gamma"

        # The topic tunnel is stored under the namespaced room.
        topical = palace_graph.follow_tunnels("wing_alpha", "topic:Angular")
        assert len(topical) == 1
        assert topical[0]["connected_wing"] == "wing_beta"

    def test_topic_tunnels_carry_kind_field(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        palace_graph.create_tunnel("wing_a", "auth", "wing_b", "users", label="x")
        palace_graph.compute_topic_tunnels({"wing_a": ["Redis"], "wing_b": ["Redis"]}, min_count=1)

        tunnels = palace_graph.list_tunnels()
        kinds = sorted(t["kind"] for t in tunnels)
        assert kinds == ["explicit", "topic"]


class TestHyphenatedWingNormalization:
    """Wing names with hyphens or spaces are normalized to underscores on init.

    Tunnel helpers must apply the same normalization at lookup time so that
    ``mempalace-public`` resolves to ``mempalace_public`` and matches the
    metadata written by ``room_detector_local.py``.
    """

    def test_list_tunnels_filters_hyphenated_wing(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel("mempalace_public", "auth", "wing_people", "users")

        assert len(palace_graph.list_tunnels("mempalace-public")) == 1
        assert len(palace_graph.list_tunnels("mempalace_public")) == 1

    def test_follow_tunnels_matches_hyphenated_wing(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        palace_graph.create_tunnel("mempalace_public", "auth", "wing_people", "users")

        by_hyphen = palace_graph.follow_tunnels("mempalace-public", "auth")
        by_under = palace_graph.follow_tunnels("mempalace_public", "auth")
        assert len(by_hyphen) == 1
        assert len(by_under) == 1
        assert by_hyphen[0]["connected_wing"] == "wing_people"

    def test_create_tunnel_normalizes_wing_names(self, tmp_path, monkeypatch):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        t = palace_graph.create_tunnel("my-project", "src", "your-project", "dst", label="cross")
        assert t["source"]["wing"] == "my_project"
        assert t["target"]["wing"] == "your_project"
        assert len(palace_graph.list_tunnels("my_project")) == 1
        assert len(palace_graph.list_tunnels("my-project")) == 1

    def test_find_tunnels_warns_on_empty_result(self, tmp_path, monkeypatch, caplog):
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        # No data in collection, so build_graph returns empty nodes
        with caplog.at_level("WARNING", logger="mempalace_graph"):
            result = palace_graph.find_tunnels("nonexistent-wing")
        assert result == []
        assert "No tunnels found" in caplog.text


class TestDrawerAwareTunnelIdentity:
    """Regressions for the 2026-05-04 tunnel-id collision bug.

    See vault planning doc 2026-05-04-mempalace-tunnel-id-audit-oddity.md.
    Identity must be drawer-aware whenever either endpoint carries a
    drawer_id; topic-level (drawer_id-less) tunnels keep room-level
    identity. Endpoint identity is as specific as the data provided.
    """

    def test_distinct_drawer_pairs_same_wing_rooms_get_distinct_ids(
        self, tmp_path, monkeypatch
    ):
        """Regression 1: multiple drawer-pair tunnels with the same source/
        target wing+room produce distinct ids and ALL persist on disk.

        Reproduces the C3 collision: two `related_distinct` ops from one
        new drawer to two distinct targets in the same wing/room got the
        same canonical id, second overwrote first."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        t1 = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            target_drawer_id="drawer_B_general_222",
            label="related",
        )
        t2 = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            target_drawer_id="drawer_B_general_333",
            label="related",
        )

        assert t1["id"] != t2["id"], (
            "two distinct drawer pairs in the same wing/room sorted to one id; "
            "drawer_id is being dropped from canonical hash"
        )
        stored = palace_graph.list_tunnels()
        assert len(stored) == 2, f"expected 2 records on disk, got {len(stored)}"
        targets_on_disk = {
            t["target"].get("drawer_id") for t in stored
        }
        assert targets_on_disk == {"drawer_B_general_222", "drawer_B_general_333"}

    def test_reversed_endpoints_same_drawer_pair_idempotent(
        self, tmp_path, monkeypatch
    ):
        """Regression 2: A→B and B→A for the SAME drawer pair must
        resolve to the same canonical id (undirected dedup preserved
        at the drawer level)."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        forward = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            target_drawer_id="drawer_B_general_222",
            label="related",
        )
        reverse = palace_graph.create_tunnel(
            "wing_b", "general", "wing_a", "do",
            source_drawer_id="drawer_B_general_222",
            target_drawer_id="drawer_A_do_111",
            label="related (updated)",
        )

        assert forward["id"] == reverse["id"]
        assert len(palace_graph.list_tunnels()) == 1
        assert reverse["label"] == "related (updated)"

    def test_hyphen_vs_underscore_wing_still_equivalent_with_drawer_ids(
        self, tmp_path, monkeypatch
    ):
        """Regression 3: wing-name normalization (hyphen → underscore)
        still applies even when drawer_ids are part of the identity.
        `vault-ops` and `vault_ops` with the same drawer pair must
        produce the same canonical id."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        hyphen = palace_graph.create_tunnel(
            "vault-ops", "do", "vault-ops", "general",
            source_drawer_id="drawer_vault_ops_do_AAA",
            target_drawer_id="drawer_vault_ops_general_BBB",
            label="x",
        )
        underscore = palace_graph.create_tunnel(
            "vault_ops", "do", "vault_ops", "general",
            source_drawer_id="drawer_vault_ops_do_AAA",
            target_drawer_id="drawer_vault_ops_general_BBB",
            label="x",
        )

        assert hyphen["id"] == underscore["id"]
        assert len(palace_graph.list_tunnels()) == 1

    def test_topic_level_tunnels_no_drawer_ids_keep_room_idempotency(
        self, tmp_path, monkeypatch
    ):
        """Regression 4a: when neither endpoint has a drawer_id (topic-level
        cross-wing links), two calls with the same wing/room pair still
        deduplicate to one record — preserving the existing topic-tunnel
        contract used by `compute_topic_tunnels`."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        first = palace_graph.create_tunnel(
            "wing_a", "topic:auth", "wing_b", "topic:auth", label="shared topic"
        )
        second = palace_graph.create_tunnel(
            "wing_a", "topic:auth", "wing_b", "topic:auth", label="updated topic"
        )

        assert first["id"] == second["id"]
        assert len(palace_graph.list_tunnels()) == 1
        assert second["label"] == "updated topic"

    def test_topic_level_does_not_collide_with_drawer_tagged(
        self, tmp_path, monkeypatch
    ):
        """Regression 4b: a drawer-tagged tunnel (any endpoint carries a
        drawer_id) must NOT collapse into a same-rooms topic-level tunnel.
        Endpoint identity is as specific as the data provided."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        topic = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general", label="topic"
        )
        tagged = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            target_drawer_id="drawer_B_general_222",
            label="tagged",
        )

        assert topic["id"] != tagged["id"]
        assert len(palace_graph.list_tunnels()) == 2

    def test_rekey_migration_updates_pre_patch_drawer_tagged_ids(
        self, tmp_path, monkeypatch
    ):
        """One-shot migration re-keys pre-patch drawer-tagged records.

        Simulate the pre-patch state by writing a tunnels.json whose
        drawer-tagged record has the OLD (room-only) canonical id.
        After ``rekey_drawer_tagged_tunnels`` runs, the record's id
        should match the drawer-aware canonical id."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        # Pre-patch id for vault_ops/dont ↔ vault_ops/general (sorted, no drawer_ids)
        old_id = hashlib.sha256(
            "vault_ops/dont↔vault_ops/general".encode()
        ).hexdigest()[:16]

        # And one topic-level record that should NOT be re-keyed.
        topic_id = hashlib.sha256(
            "wing_a/topic:auth↔wing_b/topic:auth".encode()
        ).hexdigest()[:16]

        palace_graph._save_tunnels([
            {
                "id": old_id,
                "source": {
                    "wing": "vault_ops", "room": "dont",
                    "drawer_id": "drawer_vault_ops_dont_AAA",
                },
                "target": {
                    "wing": "vault_ops", "room": "general",
                    "drawer_id": "drawer_vault_ops_general_BBB",
                },
                "label": "related",
                "kind": "explicit",
                "created_at": "2026-05-01T00:00:00+00:00",
            },
            {
                "id": topic_id,
                "source": {"wing": "wing_a", "room": "topic:auth"},
                "target": {"wing": "wing_b", "room": "topic:auth"},
                "label": "topic",
                "kind": "topic",
                "created_at": "2026-05-01T00:00:00+00:00",
            },
        ])

        report = palace_graph.rekey_drawer_tagged_tunnels()

        assert report["status"] == "ok"
        assert report["rekeyed"] == 1
        assert report["would_rekey"] == 1
        assert report["unchanged_drawer_tagged"] == 0
        assert report["topic_level_skipped"] == 1
        assert report["collisions"] == []

        new_id = palace_graph._canonical_tunnel_id(
            "vault_ops", "dont", "vault_ops", "general",
            source_drawer_id="drawer_vault_ops_dont_AAA",
            target_drawer_id="drawer_vault_ops_general_BBB",
        )
        assert new_id != old_id

        stored = palace_graph._load_tunnels()
        drawer_tagged = next(t for t in stored if t["source"].get("drawer_id"))
        topic = next(t for t in stored if not t["source"].get("drawer_id"))
        assert drawer_tagged["id"] == new_id
        assert topic["id"] == topic_id  # unchanged

        # Idempotence: a second run is a no-op.
        second = palace_graph.rekey_drawer_tagged_tunnels()
        assert second["status"] == "noop"
        assert second["rekeyed"] == 0
        assert second["would_rekey"] == 0
        assert second["unchanged_drawer_tagged"] == 1
        assert second["topic_level_skipped"] == 1

    def test_rekey_migration_refuses_to_write_on_collision(
        self, tmp_path, monkeypatch
    ):
        """Fail-closed: when two pre-patch records would re-key to the
        same drawer-aware id, the helper must NOT touch the file. The
        original (stale) ids must remain on disk so the operator can
        review labels / created_at, pick one, delete the other, then
        re-run."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        old_id = hashlib.sha256(
            "vault_ops/dont↔vault_ops/general".encode()
        ).hexdigest()[:16]
        stale_id = "deadbeefdeadbeef"
        palace_graph._save_tunnels([
            {
                "id": old_id,
                "source": {"wing": "vault_ops", "room": "dont",
                            "drawer_id": "drawer_X"},
                "target": {"wing": "vault_ops", "room": "general",
                            "drawer_id": "drawer_Y"},
                "label": "first",
                "kind": "explicit",
                "created_at": "2026-05-01T00:00:00+00:00",
            },
            {
                "id": stale_id,
                "source": {"wing": "vault_ops", "room": "dont",
                            "drawer_id": "drawer_X"},
                "target": {"wing": "vault_ops", "room": "general",
                            "drawer_id": "drawer_Y"},
                "label": "second",
                "kind": "explicit",
                "created_at": "2026-05-02T00:00:00+00:00",
            },
        ])

        report = palace_graph.rekey_drawer_tagged_tunnels()

        # Fail-closed: status reflects refusal, nothing rewritten.
        assert report["status"] == "collision_refused"
        assert report["rekeyed"] == 0
        # would_rekey reports what WOULD have been rewritten if collisions
        # were resolved first — both records here.
        assert report["would_rekey"] == 2
        assert len(report["collisions"]) == 1
        new_id = palace_graph._canonical_tunnel_id(
            "vault_ops", "dont", "vault_ops", "general",
            source_drawer_id="drawer_X",
            target_drawer_id="drawer_Y",
        )
        coll_id, idx_a, idx_b = report["collisions"][0]
        assert coll_id == new_id
        assert idx_a == 0 and idx_b == 1

        # File on disk must be untouched: original stale ids preserved.
        stored = palace_graph._load_tunnels()
        assert [t["id"] for t in stored] == [old_id, stale_id], (
            "rekey helper rewrote tunnels.json despite collision; "
            "operator cleanup is now ambiguous"
        )
        assert [t["label"] for t in stored] == ["first", "second"]

    def test_one_endpoint_drawer_tagged_other_room_only_distinct(
        self, tmp_path, monkeypatch
    ):
        """Regression 4c: asymmetric endpoint specificity is preserved.
        A tunnel with drawer_id on source only is NOT the same as one
        with drawer_ids on both — and not the same as topic-level."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        source_only = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            label="source only",
        )
        both = palace_graph.create_tunnel(
            "wing_a", "do", "wing_b", "general",
            source_drawer_id="drawer_A_do_111",
            target_drawer_id="drawer_B_general_222",
            label="both",
        )

        assert source_only["id"] != both["id"]
        assert len(palace_graph.list_tunnels()) == 2
