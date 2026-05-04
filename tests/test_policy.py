"""Tests for the write-guard policy file loader."""
import json

import pytest

from mempalace.policy import load_policy, DEFAULT_POLICY


def test_default_policy_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(tmp_path / "missing.json"))
    p = load_policy()
    assert p.guarded_writes_required is True
    assert p.default_threshold == 0.7
    assert p.advisory_floor == 0.5
    assert p.token_ttl_seconds == 300
    assert p.wings_exempt == []
    assert p.deprecated_add_drawer_response == "refuse"


def test_load_from_file(tmp_path, monkeypatch):
    pol_file = tmp_path / "policy.json"
    pol_file.write_text(json.dumps({
        "guarded_writes_required": False,
        "default_threshold": 0.65,
        "advisory_floor": 0.4,
        "token_ttl_seconds": 600,
        "wings_exempt": ["debug-wing"],
        "deprecated_add_drawer_response": "warn",
    }))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol_file))
    p = load_policy()
    assert p.guarded_writes_required is False
    assert p.default_threshold == 0.65
    assert p.wings_exempt == ["debug-wing"]
    assert p.deprecated_add_drawer_response == "warn"


def test_partial_file_uses_defaults_for_missing_keys(tmp_path, monkeypatch):
    pol_file = tmp_path / "policy.json"
    pol_file.write_text(json.dumps({"default_threshold": 0.8}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol_file))
    p = load_policy()
    assert p.default_threshold == 0.8
    # Other fields fall back to defaults
    assert p.guarded_writes_required is True
    assert p.deprecated_add_drawer_response == "refuse"


def test_invalid_deprecated_response_raises(tmp_path, monkeypatch):
    pol_file = tmp_path / "policy.json"
    pol_file.write_text(json.dumps({"deprecated_add_drawer_response": "bogus"}))
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(pol_file))
    with pytest.raises(ValueError, match="deprecated_add_drawer_response"):
        load_policy()


def test_default_policy_constant_matches_load_with_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_POLICY_PATH", str(tmp_path / "absent.json"))
    p = load_policy()
    assert p == DEFAULT_POLICY
