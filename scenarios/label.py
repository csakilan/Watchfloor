from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

from scenarios.taxonomy import ROOT_CAUSE_CATEGORIES

Specialist = Literal["preflight", "logs", "events", "config", "state"]


class Label(BaseModel):
    id: str
    incident: str
    root_cause_category: str
    root_cause_detail: str
    namespace: str
    difficulty: Literal["easy", "medium", "hard"]
    minimum_evidence: list[Specialist]
    red_herrings: list[str] = []
    source: str | None = None

    @field_validator("root_cause_category")
    @classmethod
    def category_is_in_taxonomy(cls, v: str) -> str:
        if v not in ROOT_CAUSE_CATEGORIES:
            known = ", ".join(sorted(ROOT_CAUSE_CATEGORIES))
            raise ValueError(f"'{v}' is not in the taxonomy. Known categories: {known}")
        return v


def load_label(scenario_dir: Path) -> Label:
    return Label(**yaml.safe_load((scenario_dir / "label.yaml").read_text()))


def iter_scenario_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "label.yaml").exists())
