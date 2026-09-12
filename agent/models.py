from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ModelSpec(BaseModel):
    provider: Literal["google", "ollama"]
    model: str


def load_routing(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text())
    return {role: ModelSpec(**spec) for role, spec in raw.items()}


def spec_for(role: str, routing: dict[str, ModelSpec]) -> ModelSpec:
    try:
        return routing[role]
    except KeyError:
        known = ", ".join(sorted(routing))
        raise KeyError(f"No model routing for role '{role}'. Known roles: {known}") from None
