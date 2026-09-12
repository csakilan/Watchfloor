
import pytest
from pydantic import ValidationError

from config.settings import settings
from scenarios.label import Label, iter_scenario_dirs, load_label
from scenarios.taxonomy import ROOT_CAUSE_CATEGORIES


def _valid_label_kwargs():
    return dict(
        id="001",
        incident="the checkout service is returning 503s",
        root_cause_category="missing_configmap_key",
        root_cause_detail="Deployment checkout references key DB_HOST which app-config lacks",
        namespace="shop",
        difficulty="easy",
        minimum_evidence=["config"],
    )


def test_taxonomy_has_exactly_fifteen_categories():
    assert len(ROOT_CAUSE_CATEGORIES) == 15


def test_valid_label_parses():
    label = Label(**_valid_label_kwargs())
    assert label.namespace == "shop"
    assert label.red_herrings == []


def test_category_outside_the_taxonomy_is_rejected():
    kwargs = _valid_label_kwargs() | {"root_cause_category": "gremlins"}
    with pytest.raises(ValidationError) as exc:
        Label(**kwargs)
    assert "gremlins" in str(exc.value)


def test_unknown_difficulty_is_rejected():
    kwargs = _valid_label_kwargs() | {"difficulty": "impossible"}
    with pytest.raises(ValidationError):
        Label(**kwargs)


def test_every_scenario_on_disk_has_a_valid_label():
    dirs = iter_scenario_dirs(settings.scenarios_dir)
    assert dirs, "no scenario directories found"
    for d in dirs:
        load_label(d)


def test_every_scenario_has_all_required_scripts():
    for d in iter_scenario_dirs(settings.scenarios_dir):
        for script in ("break.sh", "verify.sh", "cleanup.sh"):
            path = d / script
            assert path.exists(), f"{d.name} is missing {script}"
            assert path.stat().st_mode & 0o111, f"{d.name}/{script} is not executable"
