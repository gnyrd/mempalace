"""Tests for fate-vs-relation validation rules in commit_write."""
import pytest

from mempalace.guard import (
    validate_commit_args,
    ValidationError,
    Fate,
    Relation,
)


def _rel(drawer_id, relation, association=None):
    r = {"drawer_id": drawer_id, "relation": relation}
    if association is not None:
        r["association"] = association
    return r


# Coverage rule: every blocking candidate must be addressed exactly once
def test_missing_candidate_refused():
    with pytest.raises(ValidationError, match="not addressed"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.SUPERSEDES)],
            blocking_ids=["a", "b"],
            merged_content=None,
        )


def test_extra_candidate_refused():
    with pytest.raises(ValidationError, match="not in blocking"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.SUPERSEDES), _rel("c", Relation.IGNORED_SPURIOUS)],
            blocking_ids=["a"],
            merged_content=None,
        )


def test_duplicate_candidate_classification_refused():
    with pytest.raises(ValidationError, match="addressed twice"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.SUPERSEDES), _rel("a", Relation.IGNORED_SPURIOUS)],
            blocking_ids=["a"],
            merged_content=None,
        )


# Fate-vs-relation rules
def test_create_new_with_duplicate_relation_refused():
    with pytest.raises(ValidationError, match="duplicate.*not allowed.*create_new"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.DUPLICATE)],
            blocking_ids=["a"],
            merged_content=None,
        )


def test_create_new_with_strengthens_relation_refused():
    with pytest.raises(ValidationError, match="strengthens_target.*not allowed.*create_new"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.STRENGTHENS_TARGET)],
            blocking_ids=["a"],
            merged_content="anything",
        )


def test_skip_new_requires_at_least_one_duplicate():
    with pytest.raises(ValidationError, match="skip_new requires.*duplicate"):
        validate_commit_args(
            fate=Fate.SKIP_NEW,
            candidate_relations=[_rel("a", Relation.IGNORED_SPURIOUS)],
            blocking_ids=["a"],
            merged_content=None,
        )


def test_skip_new_with_supersedes_refused():
    with pytest.raises(ValidationError, match="skip_new.*only.*duplicate.*ignored_spurious"):
        validate_commit_args(
            fate=Fate.SKIP_NEW,
            candidate_relations=[_rel("a", Relation.DUPLICATE), _rel("b", Relation.SUPERSEDES)],
            blocking_ids=["a", "b"],
            merged_content=None,
        )


def test_skip_new_with_merged_content_refused():
    with pytest.raises(ValidationError, match="merged_content.*absent.*skip_new"):
        validate_commit_args(
            fate=Fate.SKIP_NEW,
            candidate_relations=[_rel("a", Relation.DUPLICATE)],
            blocking_ids=["a"],
            merged_content="oops",
        )


def test_update_existing_requires_one_strengthens():
    with pytest.raises(ValidationError, match="update_existing requires.*strengthens_target"):
        validate_commit_args(
            fate=Fate.UPDATE_EXISTING,
            candidate_relations=[_rel("a", Relation.IGNORED_SPURIOUS)],
            blocking_ids=["a"],
            merged_content="merged",
        )


def test_update_existing_two_strengthens_refused():
    with pytest.raises(ValidationError, match="exactly one strengthens_target"):
        validate_commit_args(
            fate=Fate.UPDATE_EXISTING,
            candidate_relations=[_rel("a", Relation.STRENGTHENS_TARGET), _rel("b", Relation.STRENGTHENS_TARGET)],
            blocking_ids=["a", "b"],
            merged_content="merged",
        )


def test_update_existing_without_merged_content_refused():
    with pytest.raises(ValidationError, match="merged_content.*required"):
        validate_commit_args(
            fate=Fate.UPDATE_EXISTING,
            candidate_relations=[_rel("a", Relation.STRENGTHENS_TARGET)],
            blocking_ids=["a"],
            merged_content=None,
        )


# related_distinct requires association
def test_related_distinct_without_association_refused():
    with pytest.raises(ValidationError, match="related_distinct.*requires association"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[_rel("a", Relation.RELATED_DISTINCT)],
            blocking_ids=["a"],
            merged_content=None,
        )


# Happy paths
def test_create_new_with_supersedes_passes():
    validate_commit_args(
        fate=Fate.CREATE_NEW,
        candidate_relations=[_rel("a", Relation.SUPERSEDES)],
        blocking_ids=["a"],
        merged_content=None,
    )


def test_create_new_mixed_relations_passes():
    validate_commit_args(
        fate=Fate.CREATE_NEW,
        candidate_relations=[
            _rel("a", Relation.SUPERSEDES),
            _rel("b", Relation.RELATED_DISTINCT, {"kind": "tunnel", "direction": "to_new"}),
            _rel("c", Relation.IGNORED_SPURIOUS),
        ],
        blocking_ids=["a", "b", "c"],
        merged_content=None,
    )


def test_skip_new_with_duplicate_and_ignored_passes():
    validate_commit_args(
        fate=Fate.SKIP_NEW,
        candidate_relations=[_rel("a", Relation.DUPLICATE), _rel("b", Relation.IGNORED_SPURIOUS)],
        blocking_ids=["a", "b"],
        merged_content=None,
    )


def test_update_existing_passes():
    validate_commit_args(
        fate=Fate.UPDATE_EXISTING,
        candidate_relations=[_rel("a", Relation.STRENGTHENS_TARGET), _rel("b", Relation.IGNORED_SPURIOUS)],
        blocking_ids=["a", "b"],
        merged_content="merged content here",
    )


# Shape validation: malformed candidate_relations entries must surface as
# typed ValidationError rather than bare KeyError. Important because Task 9
# will wire validate_commit_args into tool_commit_write, where bare
# KeyError would surface as an unstructured MCP error. (Codex v2.1 review.)


def test_missing_drawer_id_raises_validation_error():
    with pytest.raises(ValidationError, match="missing required key 'drawer_id'"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[{"relation": Relation.SUPERSEDES}],
            blocking_ids=["a"],
            merged_content=None,
        )


def test_missing_relation_raises_validation_error():
    with pytest.raises(ValidationError, match="missing required key 'relation'"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=[{"drawer_id": "a"}],
            blocking_ids=["a"],
            merged_content=None,
        )


def test_non_dict_candidate_entry_raises_validation_error():
    with pytest.raises(ValidationError, match="must be a dict"):
        validate_commit_args(
            fate=Fate.CREATE_NEW,
            candidate_relations=["not a dict"],
            blocking_ids=["a"],
            merged_content=None,
        )
