"""Write-guard policy loader.

Reads a JSON policy file from MEMPALACE_POLICY_PATH (default ~/.mempalace/policy.json)
and returns a typed Policy. Missing keys fall back to defaults; missing files use
the entire default policy. The guard imports `load_policy()` at startup and on
explicit reload.
"""
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

_VALID_DEPRECATED_RESPONSES = {"refuse", "warn", "allow"}


@dataclass(frozen=True)
class Policy:
    guarded_writes_required: bool = True
    default_threshold: float = 0.7
    advisory_floor: float = 0.5
    token_ttl_seconds: int = 300
    wings_exempt: list = field(default_factory=list)
    deprecated_add_drawer_response: str = "refuse"


DEFAULT_POLICY = Policy()


def _policy_path() -> Path:
    return Path(os.environ.get(
        "MEMPALACE_POLICY_PATH",
        os.path.expanduser("~/.mempalace/policy.json"),
    ))


def load_policy() -> Policy:
    """Read policy file; merge with defaults; validate; return a Policy."""
    path = _policy_path()
    if not path.exists():
        return DEFAULT_POLICY

    raw = json.loads(path.read_text())

    if "deprecated_add_drawer_response" in raw:
        v = raw["deprecated_add_drawer_response"]
        if v not in _VALID_DEPRECATED_RESPONSES:
            raise ValueError(
                f"deprecated_add_drawer_response must be one of "
                f"{sorted(_VALID_DEPRECATED_RESPONSES)}, got {v!r}"
            )

    # Merge raw onto defaults
    overrides = {k: v for k, v in raw.items() if k in Policy.__dataclass_fields__}
    return replace(DEFAULT_POLICY, **overrides)
