import pytest

from config.settings import REPO_ROOT
from agent.models import ModelSpec, load_routing, spec_for


def test_load_routing_parses_all_four_roles():
    routing = load_routing(REPO_ROOT / "config" / "models.yaml")
    assert set(routing) == {"triage", "supervisor", "specialists", "synthesizer"}
    assert all(isinstance(v, ModelSpec) for v in routing.values())


def test_spec_for_returns_the_spec():
    routing = {"supervisor": ModelSpec(provider="google", model="test-model-a")}
    assert spec_for("supervisor", routing).model == "test-model-a"


def test_spec_for_unknown_role_lists_the_known_roles():
    routing = {"supervisor": ModelSpec(provider="google", model="test-model-a")}
    with pytest.raises(KeyError) as exc:
        spec_for("librarian", routing)
    assert "librarian" in str(exc.value)
    assert "supervisor" in str(exc.value)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        ModelSpec(provider="openai", model="whatever")
