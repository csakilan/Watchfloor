# Watchfloor Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project foundations and answer the single riskiest open question in the design, which is whether a Flash-tier model can reliably choose a question template and fill its slots.

**Architecture:** Six tasks. The first four are pure Python and shell with no model calls and no cluster required for their unit tests: project skeleton and model routing, the scenario label contract plus the first seeded scenario, kind cluster management, and a scenario script runner. The fifth adds a rate-limited model client. The sixth runs the capability spike and records its result in `DECISIONS.md`. Every task ends with a commit.

**Tech Stack:** Python 3.12, Pydantic v2, pydantic-settings, PyYAML, pytest, kind, kubectl, google-genai.

## Global Constraints

- Python 3.12.
- No write operations against the cluster from any agent-facing code. Cluster mutation exists only in `scenarios/*/break.sh` and `scenarios/*/cleanup.sh`, which are test fixtures, never agent tools.
- No model name appears anywhere outside `config/models.yaml`.
- Prompts and question templates live in files, never string literals.
- Every model output that drives control flow is schema-constrained. Never parse free text for a decision.
- Iteration caps are never raised to make something pass.
- `label.yaml` is never edited to match what a system produced.
- Tool argument schemas are flat: primitives and string enums only, no nested objects.
- Tests that need a live cluster are marked `@pytest.mark.integration` and excluded from the default run.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest configuration |
| `config/settings.py` | Environment-derived settings, one `Settings` class |
| `config/models.yaml` | Model routing by node role. The only place model names appear |
| `agent/models.py` | Loads and validates model routing, resolves a role to a `ModelSpec` |
| `agent/llm.py` | Rate limiting, retry, request counting, provider adapter |
| `agent/templates/questions.yaml` | Slot-fill question templates, keyed by specialist |
| `agent/questions.py` | Loads templates, validates a decision against them |
| `cluster/manage.py` | Idempotent kind cluster create, delete, existence check |
| `scenarios/taxonomy.py` | The fixed fifteen root-cause categories |
| `scenarios/label.py` | `Label` schema and loader, validates against the taxonomy |
| `scenarios/runner.py` | Executes a scenario's break, verify and cleanup scripts |
| `scenarios/001-missing-configmap-key/` | First seeded scenario, five files |
| `spikes/slot_fill.py` | The Phase 0 capability spike, writes JSON to `results/` |
| `DECISIONS.md` | One line per decision, seeded with the six design deltas |

---

## Task 1: Project skeleton, settings, and model routing

**Files:**
- Create: `pyproject.toml`
- Create: `config/__init__.py`, `config/settings.py`, `config/models.yaml`
- Create: `agent/__init__.py`, `agent/models.py`
- Create: `DECISIONS.md`, `README.md`, `.gitignore`
- Test: `tests/test_model_routing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.settings.Settings`, `config.settings.REPO_ROOT`, `agent.models.ModelSpec` with fields `provider: Literal["google","ollama"]` and `model: str`, `agent.models.load_routing(path: Path) -> dict[str, ModelSpec]`, `agent.models.spec_for(role: str, routing: dict[str, ModelSpec]) -> ModelSpec`.

- [ ] **Step 1: Create the package skeleton and dependency manifest**

```bash
mkdir -p config agent/templates cluster scenarios spikes eval results tests
touch config/__init__.py agent/__init__.py cluster/__init__.py scenarios/__init__.py
```

`pyproject.toml`:

```toml
[project]
name = "watchfloor"
version = "0.1.0"
description = "Multi-agent Kubernetes incident diagnosis with a measured evaluation harness"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pyyaml>=6.0",
    "google-genai>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["agent", "config", "cluster", "scenarios"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires a live kind cluster, excluded from the default run",
]
addopts = "-m 'not integration'"
```

`.gitignore`:

```
__pycache__/
*.egg-info/
.venv/
.env
```

- [ ] **Step 2: Write the failing test**

`tests/test_model_routing.py`:

```python

import pytest

from config.settings import REPO_ROOT
from agent.models import ModelSpec, load_routing, spec_for


def test_load_routing_parses_all_four_roles():
    routing = load_routing(REPO_ROOT / "config" / "models.yaml")
    assert set(routing) == {"triage", "supervisor", "specialists", "synthesizer"}
    assert all(isinstance(v, ModelSpec) for v in routing.values())


def test_spec_for_returns_the_spec():
    routing = {"supervisor": ModelSpec(provider="google", model="gemini-2.5-flash")}
    assert spec_for("supervisor", routing).model == "gemini-2.5-flash"


def test_spec_for_unknown_role_lists_the_known_roles():
    routing = {"supervisor": ModelSpec(provider="google", model="gemini-2.5-flash")}
    with pytest.raises(KeyError) as exc:
        spec_for("librarian", routing)
    assert "librarian" in str(exc.value)
    assert "supervisor" in str(exc.value)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        ModelSpec(provider="openai", model="whatever")
```

The third test matters more than it looks. It is the same principle as the MCP server's recoverable errors: a failure should carry the information needed to fix it.

- [ ] **Step 3: Run the test and confirm it fails**

Run: `pytest tests/test_model_routing.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'agent.models'`

- [ ] **Step 4: Write `config/models.yaml`**

```yaml
triage:      { provider: google, model: gemini-2.5-flash }
supervisor:  { provider: google, model: gemini-2.5-flash }
specialists: { provider: google, model: gemini-2.5-flash-lite }
synthesizer: { provider: google, model: gemini-2.5-flash }
```

- [ ] **Step 5: Write `config/settings.py`**

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WATCHFLOOR_", env_file=".env", extra="ignore"
    )

    google_api_key: str = ""
    cluster_name: str = "watchfloor"
    requests_per_minute: int = 12
    model_routing_path: Path = REPO_ROOT / "config" / "models.yaml"
    scenarios_dir: Path = REPO_ROOT / "scenarios"
    results_dir: Path = REPO_ROOT / "results"


settings = Settings()
```

`requests_per_minute` defaults to 12 rather than 15 to leave headroom under the free-tier limit.

- [ ] **Step 6: Write `agent/models.py`**

```python
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
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `pytest tests/test_model_routing.py -v`
Expected: 4 passed

- [ ] **Step 8: Write `DECISIONS.md` seeded with the design deltas**

```markdown
# Decisions

One line per decision: what changed, and why.

## 2026-09-10, from the design document

- Supervisor emits a template choice plus slot values instead of a free-text question. Free generation is unreliable at Flash tier.
- Synthesizer emits an ordinal confidence level; the numeric score is derived deterministically. Small models are badly calibrated and confident-wrong is the headline metric.
- Graph starts at a deterministic `preflight` node before `triage`. Enumerable work belongs in code.
- Added a Phase 0 capability spike. The weak-model adaptation is an untested bet and testing it in week one is cheap.
- Evaluation adds label pre-registration, a five-scenario held-out set, and external anchoring. Publishing the suite alone does not address overfitting during tuning.
- The single-agent baseline is also built on LangGraph, so the comparison differs in graph shape rather than in codebase.
- Default request rate set to 12 per minute, below the observed free-tier ceiling, to leave headroom for retries.
```

- [ ] **Step 9: Write a `README.md` stub**

```markdown
# Watchfloor

Multi-agent Kubernetes incident diagnosis with a measured evaluation harness.

Status: Phase 0, foundations. Nothing to run yet beyond the tests.

Design: `docs/superpowers/specs/2026-09-10-watchfloor-design.md`
Decisions: `DECISIONS.md`
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore README.md DECISIONS.md config/ agent/ tests/
git commit -m "feat: project skeleton, settings, and model routing"
```

---

## Task 2: Taxonomy, label contract, and the first scenario

**Files:**
- Create: `scenarios/taxonomy.py`, `scenarios/label.py`
- Create: `scenarios/001-missing-configmap-key/manifest.yaml`
- Create: `scenarios/001-missing-configmap-key/break.sh`, `verify.sh`, `cleanup.sh`, `label.yaml`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `config.settings.settings.scenarios_dir`.
- Produces: `scenarios.taxonomy.ROOT_CAUSE_CATEGORIES: frozenset[str]`, `scenarios.label.Label`, `scenarios.label.load_label(scenario_dir: Path) -> Label`, `scenarios.label.iter_scenario_dirs(root: Path) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:

```python

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
```

The last two tests run over whatever is on disk, so they keep guarding every scenario added in later phases without anyone remembering to extend them.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scenarios.taxonomy'`

- [ ] **Step 3: Write `scenarios/taxonomy.py`**

```python
ROOT_CAUSE_CATEGORIES: frozenset[str] = frozenset(
    {
        "bad_image_tag",
        "image_pull_secret_missing",
        "oom_killed",
        "missing_configmap_key",
        "missing_secret",
        "wrong_service_selector",
        "readiness_probe_wrong_port",
        "liveness_probe_too_aggressive",
        "networkpolicy_blocking_traffic",
        "pvc_pending_no_storageclass",
        "resource_quota_exceeded",
        "node_selector_unschedulable",
        "crashloop_bad_env_var",
        "init_container_failing",
        "service_port_mismatch",
    }
)
```

- [ ] **Step 4: Write `scenarios/label.py`**

```python
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
```

`source` is the external-anchoring field from the design: where this failure mode was documented in the real world.

- [ ] **Step 5: Write the scenario manifest**

`scenarios/001-missing-configmap-key/manifest.yaml`. This is the healthy baseline, so `DB_HOST` is present here and `break.sh` removes it. The `inventory-cache` deployment is the red herring: it exits cleanly every fifteen seconds and therefore accumulates restarts for a reason that has nothing to do with the fault.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: shop
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: shop
data:
  DB_HOST: postgres.shop.svc.cluster.local
  LOG_LEVEL: info
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
  namespace: shop
spec:
  replicas: 1
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
        - name: checkout
          image: busybox:1.36
          command: ["sh", "-c", "echo starting against $DB_HOST; sleep 3600"]
          env:
            - name: DB_HOST
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: DB_HOST
---
apiVersion: v1
kind: Service
metadata:
  name: checkout
  namespace: shop
spec:
  selector:
    app: checkout
  ports:
    - port: 8080
      targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inventory-cache
  namespace: shop
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inventory-cache
  template:
    metadata:
      labels:
        app: inventory-cache
    spec:
      containers:
        - name: cache
          image: busybox:1.36
          command: ["sh", "-c", "sleep 15"]
```

- [ ] **Step 6: Write the three scripts**

`break.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl apply -f "$DIR/manifest.yaml"
kubectl -n shop rollout status deployment/checkout --timeout=90s

kubectl -n shop patch configmap app-config \
  --type=json -p='[{"op": "remove", "path": "/data/DB_HOST"}]'
kubectl -n shop rollout restart deployment/checkout
```

`verify.sh`. Exit 0 means the fault is live as intended:

```bash
#!/usr/bin/env bash
set -uo pipefail

for _ in $(seq 1 30); do
  reasons=$(kubectl -n shop get pods -l app=checkout \
    -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null)
  if echo "$reasons" | grep -q CreateContainerConfigError; then
    echo "fault confirmed: checkout pod in CreateContainerConfigError"
    exit 0
  fi
  sleep 2
done

echo "fault NOT confirmed after 60s. Observed waiting reasons: ${reasons:-none}" >&2
exit 1
```

`cleanup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
kubectl delete namespace shop --ignore-not-found --wait=true
```

Then make them executable:

```bash
chmod +x scenarios/001-missing-configmap-key/*.sh
```

- [ ] **Step 7: Write `label.yaml`**

```yaml
id: "001"
incident: "the checkout service is returning 503s"
root_cause_category: missing_configmap_key
root_cause_detail: >-
  Deployment checkout in namespace shop reads DB_HOST via configMapKeyRef from
  configmap app-config, which no longer contains that key. The container cannot
  be created, so the Service has no ready endpoints.
namespace: shop
difficulty: easy
minimum_evidence:
  - config
red_herrings:
  - "inventory-cache in the same namespace restarts every 15 seconds for an unrelated and benign reason"
source: "Common Kubernetes failure mode: configMapKeyRef to a key removed by a later edit"
```

- [ ] **Step 8: Run the tests and confirm they pass**

Run: `pytest tests/test_labels.py -v`
Expected: 6 passed

- [ ] **Step 9: Commit, and note that this commit is the pre-registration**

```bash
git add scenarios/ tests/test_labels.py
git commit -m "feat: taxonomy, label contract, and scenario 001 (label pre-registration)"
```

The design relies on git history proving labels were written before any agent ran against them. This commit is that proof for scenario 001, so it must land before any agent code exists.

---

## Task 3: Idempotent kind cluster management

**Files:**
- Create: `cluster/manage.py`, `cluster/kind-config.yaml`
- Test: `tests/test_cluster_manage.py`

**Interfaces:**
- Consumes: `config.settings.settings.cluster_name`.
- Produces: `cluster.manage.list_clusters(runner=...) -> list[str]`, `cluster.manage.cluster_exists(name, runner=...) -> bool`, `cluster.manage.create_cluster(name, config_path=None, runner=...) -> bool` returning True when it created and False when it already existed, `cluster.manage.delete_cluster(name, runner=...) -> bool`.

A `runner` parameter is injected so unit tests never shell out. Default is `subprocess.run`.

- [ ] **Step 1: Write the failing test**

`tests/test_cluster_manage.py`:

```python
import subprocess
from dataclasses import dataclass, field

import pytest

from cluster.manage import cluster_exists, create_cluster, delete_cluster, list_clusters


@dataclass
class FakeRunner:
    stdout: str = ""
    returncode: int = 0
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "")


def test_list_clusters_splits_lines():
    runner = FakeRunner(stdout="watchfloor\nother\n")
    assert list_clusters(runner=runner) == ["watchfloor", "other"]


def test_list_clusters_handles_no_clusters():
    assert list_clusters(runner=FakeRunner(stdout="\n")) == []


def test_cluster_exists_is_true_when_named():
    assert cluster_exists("watchfloor", runner=FakeRunner(stdout="watchfloor\n")) is True


def test_create_is_a_noop_when_the_cluster_exists():
    runner = FakeRunner(stdout="watchfloor\n")
    assert create_cluster("watchfloor", runner=runner) is False
    assert len(runner.calls) == 1, "should not have called kind create"


def test_create_runs_kind_create_when_absent():
    runner = FakeRunner(stdout="")
    assert create_cluster("watchfloor", runner=runner) is True
    assert runner.calls[-1][:3] == ["kind", "create", "cluster"]
    assert "watchfloor" in runner.calls[-1]


def test_delete_is_a_noop_when_absent():
    runner = FakeRunner(stdout="")
    assert delete_cluster("watchfloor", runner=runner) is False


@pytest.mark.integration
def test_create_then_create_again_is_idempotent():
    from config.settings import settings

    create_cluster(settings.cluster_name)
    assert create_cluster(settings.cluster_name) is False
    assert cluster_exists(settings.cluster_name) is True
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_cluster_manage.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'cluster.manage'`

- [ ] **Step 3: Write `cluster/kind-config.yaml`**

A single control-plane node is enough for every scenario except `node_selector_unschedulable`, which needs somewhere it cannot schedule, so two workers are provisioned now to avoid recreating the cluster in Phase 1.

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

- [ ] **Step 4: Write `cluster/manage.py`**

```python
import subprocess
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "kind-config.yaml"


def _run(cmd: list[str], runner=subprocess.run) -> subprocess.CompletedProcess:
    return runner(cmd, capture_output=True, text=True, check=False)


def list_clusters(runner=subprocess.run) -> list[str]:
    result = _run(["kind", "get", "clusters"], runner=runner)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cluster_exists(name: str, runner=subprocess.run) -> bool:
    return name in list_clusters(runner=runner)


def create_cluster(name: str, config_path: Path | None = None, runner=subprocess.run) -> bool:
    if cluster_exists(name, runner=runner):
        return False
    config = config_path or DEFAULT_CONFIG
    result = _run(
        ["kind", "create", "cluster", "--name", name, "--config", str(config)], runner=runner
    )
    if result.returncode != 0:
        raise RuntimeError(f"kind create cluster failed:\n{result.stderr}")
    return True


def delete_cluster(name: str, runner=subprocess.run) -> bool:
    if not cluster_exists(name, runner=runner):
        return False
    result = _run(["kind", "delete", "cluster", "--name", name], runner=runner)
    if result.returncode != 0:
        raise RuntimeError(f"kind delete cluster failed:\n{result.stderr}")
    return True
```

- [ ] **Step 5: Run the unit tests and confirm they pass**

Run: `pytest tests/test_cluster_manage.py -v`
Expected: 6 passed, 1 deselected

- [ ] **Step 6: Run the integration test once by hand**

Run: `pytest tests/test_cluster_manage.py -v -m integration`
Expected: PASS. First run takes a minute or two while kind pulls node images. Confirm with `kubectl config current-context`, which should print `kind-watchfloor`.

- [ ] **Step 7: Commit**

```bash
git add cluster/ tests/test_cluster_manage.py
git commit -m "feat: idempotent kind cluster management"
```

---

## Task 4: Scenario script runner

**Files:**
- Create: `scenarios/runner.py`
- Test: `tests/test_scenario_runner.py`

**Interfaces:**
- Consumes: `scenarios.label.load_label`.
- Produces: `scenarios.runner.ScriptResult` with fields `script: str`, `exit_code: int`, `stdout: str`, `stderr: str`, `duration_s: float`; and `scenarios.runner.run_script(scenario_dir: Path, script: str, timeout: int = 180) -> ScriptResult`, plus `break_scenario`, `verify_scenario`, `cleanup_scenario`, each taking `scenario_dir: Path` and returning `ScriptResult`.

- [ ] **Step 1: Write the failing test**

`tests/test_scenario_runner.py`:

```python
import subprocess
from pathlib import Path

import pytest

from scenarios.runner import run_script, verify_scenario


def _write_script(d: Path, name: str, body: str) -> None:
    path = d / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def test_captures_stdout_and_zero_exit(tmp_path):
    _write_script(tmp_path, "break.sh", "echo broke it")
    result = run_script(tmp_path, "break.sh")
    assert result.exit_code == 0
    assert "broke it" in result.stdout
    assert result.duration_s >= 0


def test_captures_nonzero_exit_and_stderr(tmp_path):
    _write_script(tmp_path, "verify.sh", "echo nope >&2; exit 1")
    result = verify_scenario(tmp_path)
    assert result.exit_code == 1
    assert "nope" in result.stderr


def test_missing_script_names_what_is_present(tmp_path):
    _write_script(tmp_path, "break.sh", "true")
    with pytest.raises(FileNotFoundError) as exc:
        run_script(tmp_path, "verify.sh")
    assert "verify.sh" in str(exc.value)
    assert "break.sh" in str(exc.value)


def test_timeout_raises(tmp_path):
    _write_script(tmp_path, "break.sh", "sleep 5")
    with pytest.raises(subprocess.TimeoutExpired):
        run_script(tmp_path, "break.sh", timeout=1)
```

The third test applies the recoverable-error principle to the harness itself.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_scenario_runner.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scenarios.runner'`

- [ ] **Step 3: Write `scenarios/runner.py`**

```python
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScriptResult:
    script: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


def run_script(scenario_dir: Path, script: str, timeout: int = 180) -> ScriptResult:
    path = scenario_dir / script
    if not path.exists():
        present = ", ".join(sorted(p.name for p in scenario_dir.glob("*.sh"))) or "none"
        raise FileNotFoundError(
            f"'{script}' not found in {scenario_dir}. Scripts present: {present}"
        )

    started = time.monotonic()
    completed = subprocess.run(
        [str(path)], capture_output=True, text=True, timeout=timeout, check=False
    )
    return ScriptResult(
        script=script,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=time.monotonic() - started,
    )


def break_scenario(scenario_dir: Path) -> ScriptResult:
    return run_script(scenario_dir, "break.sh")


def verify_scenario(scenario_dir: Path) -> ScriptResult:
    return run_script(scenario_dir, "verify.sh")


def cleanup_scenario(scenario_dir: Path) -> ScriptResult:
    return run_script(scenario_dir, "cleanup.sh")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_scenario_runner.py -v`
Expected: 4 passed

- [ ] **Step 5: Add the end-to-end integration test**

Append to `tests/test_scenario_runner.py`:

```python
@pytest.mark.integration
def test_scenario_001_breaks_and_verifies():
    from cluster.manage import create_cluster
    from config.settings import settings
    from scenarios.runner import break_scenario, cleanup_scenario

    create_cluster(settings.cluster_name)
    scenario = settings.scenarios_dir / "001-missing-configmap-key"
    try:
        assert break_scenario(scenario).exit_code == 0
        assert verify_scenario(scenario).exit_code == 0
    finally:
        cleanup_scenario(scenario)
```

- [ ] **Step 6: Run it and confirm the fault is real**

Run: `pytest tests/test_scenario_runner.py -v -m integration`
Expected: PASS. This is the Phase 0 acceptance gate for the scenario contract, and it proves `break.sh` produces the fault that `verify.sh` claims.

- [ ] **Step 7: Commit**

```bash
git add scenarios/runner.py tests/test_scenario_runner.py
git commit -m "feat: scenario script runner with break/verify/cleanup"
```

---

## Task 5: Rate-limited, counted model client

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `agent.models.ModelSpec`, `config.settings.settings`.
- Produces: `agent.llm.RateLimiter(rpm: int, clock=..., sleep=...)` with method `acquire() -> None`; `agent.llm.RequestCounter` with `total: int`, `by_role: dict[str, int]`, and `record(role: str) -> None`; `agent.llm.generate_structured(prompt: str, schema: type[BaseModel], role: str, *, spec: ModelSpec, limiter: RateLimiter, counter: RequestCounter, call=...) -> BaseModel`.

The `call` parameter is the provider adapter, injected so every test in this task runs offline.

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:

```python
import pytest
from pydantic import BaseModel

from agent.llm import RateLimiter, RequestCounter, generate_structured
from agent.models import ModelSpec


class Answer(BaseModel):
    verdict: str


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_limiter_does_not_sleep_below_the_rate():
    clock, slept = FakeClock(), []
    limiter = RateLimiter(rpm=60, clock=clock, sleep=slept.append)
    limiter.acquire()
    clock.now = 1.0
    limiter.acquire()
    assert slept == []


def test_limiter_sleeps_when_calls_are_too_close():
    clock, slept = FakeClock(), []
    limiter = RateLimiter(rpm=60, clock=clock, sleep=slept.append)
    limiter.acquire()
    limiter.acquire()
    assert slept and slept[0] == pytest.approx(1.0)


def test_counter_tracks_total_and_role():
    counter = RequestCounter()
    counter.record("supervisor")
    counter.record("supervisor")
    counter.record("logs")
    assert counter.total == 3
    assert counter.by_role == {"supervisor": 2, "logs": 1}


def test_generate_structured_returns_the_parsed_schema_and_counts():
    counter = RequestCounter()
    limiter = RateLimiter(rpm=6000, clock=FakeClock(), sleep=lambda _: None)
    result = generate_structured(
        prompt="anything",
        schema=Answer,
        role="supervisor",
        spec=ModelSpec(provider="google", model="gemini-2.5-flash"),
        limiter=limiter,
        counter=counter,
        call=lambda **kw: '{"verdict": "ok"}',
    )
    assert result.verdict == "ok"
    assert counter.total == 1


def test_generate_structured_raises_on_unparseable_output():
    counter = RequestCounter()
    limiter = RateLimiter(rpm=6000, clock=FakeClock(), sleep=lambda _: None)
    with pytest.raises(ValueError) as exc:
        generate_structured(
            prompt="anything",
            schema=Answer,
            role="supervisor",
            spec=ModelSpec(provider="google", model="gemini-2.5-flash"),
            limiter=limiter,
            counter=counter,
            call=lambda **kw: "I think the answer is probably fine",
        )
    assert "supervisor" in str(exc.value)
```

The last test pins down behaviour the design depends on: unparseable output raises rather than being coerced, so Phase 3 can route to a deterministic fallback instead of silently degrading.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Write `agent/llm.py`**

```python
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from agent.models import ModelSpec


class RateLimiter:
    """Spaces calls to at most `rpm` per minute."""

    def __init__(self, rpm: int, clock=time.monotonic, sleep=time.sleep):
        self._interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def acquire(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self._interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
                now = now + wait
        self._last = now


@dataclass
class RequestCounter:
    total: int = 0
    by_role: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, role: str) -> None:
        self.total += 1
        self.by_role[role] += 1


def _google_call(*, prompt: str, spec: ModelSpec, schema: type[BaseModel]) -> str:
    from google import genai

    from config.settings import settings

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=spec.model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema.model_json_schema(),
        },
    )
    return response.text


def generate_structured(
    prompt: str,
    schema: type[BaseModel],
    role: str,
    *,
    spec: ModelSpec,
    limiter: RateLimiter,
    counter: RequestCounter,
    call=_google_call,
) -> BaseModel:
    limiter.acquire()
    counter.record(role)
    raw = call(prompt=prompt, spec=spec, schema=schema)
    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Model output for role '{role}' did not match {schema.__name__}. "
            f"Raw output: {raw[:400]}"
        ) from exc
```

`by_role` uses a `defaultdict`, which compares equal to a plain dict, so the test assertion holds.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_llm.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify the provider SDK against a live call**

Set `WATCHFLOOR_GOOGLE_API_KEY` in `.env`, then:

```bash
python -c "
from agent.llm import RateLimiter, RequestCounter, generate_structured
from agent.models import load_routing, spec_for
from config.settings import settings
from pydantic import BaseModel

class Ping(BaseModel):
    answer: str

routing = load_routing(settings.model_routing_path)
print(generate_structured(
    prompt='Reply with the JSON object {\"answer\": \"pong\"} and nothing else.',
    schema=Ping, role='supervisor',
    spec=spec_for('supervisor', routing),
    limiter=RateLimiter(settings.requests_per_minute),
    counter=RequestCounter(),
))
"
```

Expected: `answer='pong'`. If the import of `google.genai` or the config keys fail, the SDK surface has changed since this plan was written. Fix `_google_call` only, leave everything else alone, and record the change in `DECISIONS.md`.

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: rate-limited model client with request counting"
```

---

## Task 6: The slot-fill capability spike

This is the reason Phase 0 exists. Everything above is scaffolding for this measurement.

**Files:**
- Create: `agent/templates/questions.yaml`, `agent/questions.py`
- Create: `spikes/__init__.py`, `spikes/slot_fill.py`
- Test: `tests/test_templates.py`

**Interfaces:**
- Consumes: `agent.llm.generate_structured`, `agent.models.load_routing`, `agent.models.spec_for`, `config.settings.settings`.
- Produces: `agent.questions.QuestionTemplate` with fields `id: str`, `template: str`, `slots: list[str]`; `agent.questions.load_templates(path: Path) -> dict[str, list[QuestionTemplate]]`; `agent.questions.SlotFillDecision` with fields `specialist: str`, `template_id: str`, `slot_values: list[str]`; `agent.questions.validate_decision(decision, templates) -> list[str]` returning a list of problem strings, empty when valid; `agent.questions.render(decision, templates) -> str`.

`slot_values` is a flat list positionally matching the template's declared slots. A nested object would be a harder ask of a small model, and the whole point of this task is to avoid that.

- [ ] **Step 1: Write the failing test**

`tests/test_templates.py`:

```python

import pytest

from config.settings import REPO_ROOT
from agent.questions import (
    QuestionTemplate,
    SlotFillDecision,
    load_templates,
    render,
    validate_decision,
)

TEMPLATES = {
    "config": [
        QuestionTemplate(
            id="cfg_missing_ref",
            template="does {kind}/{name} in namespace {namespace} reference a missing {resource_kind}?",
            slots=["kind", "name", "namespace", "resource_kind"],
        )
    ]
}


def _decision(**overrides) -> SlotFillDecision:
    base = dict(
        specialist="config",
        template_id="cfg_missing_ref",
        slot_values=["Deployment", "checkout", "shop", "ConfigMap"],
    )
    return SlotFillDecision(**(base | overrides))


def test_valid_decision_has_no_problems():
    assert validate_decision(_decision(), TEMPLATES) == []


def test_unknown_specialist_is_a_problem():
    problems = validate_decision(_decision(specialist="logs"), TEMPLATES)
    assert any("logs" in p for p in problems)


def test_unknown_template_id_is_a_problem():
    problems = validate_decision(_decision(template_id="cfg_nope"), TEMPLATES)
    assert any("cfg_nope" in p for p in problems)


def test_wrong_slot_count_is_a_problem():
    problems = validate_decision(_decision(slot_values=["Deployment", "checkout"]), TEMPLATES)
    assert any("4" in p and "2" in p for p in problems)


def test_placeholder_slot_values_are_problems():
    for junk in ("", "  ", "unknown", "TODO", "kind", "{kind}"):
        problems = validate_decision(
            _decision(slot_values=[junk, "checkout", "shop", "ConfigMap"]), TEMPLATES
        )
        assert problems, f"{junk!r} should have been rejected"


def test_render_substitutes_positionally():
    assert render(_decision(), TEMPLATES) == (
        "does Deployment/checkout in namespace shop reference a missing ConfigMap?"
    )


def test_shipped_templates_all_declare_slots_that_appear_in_their_string():
    templates = load_templates(REPO_ROOT / "agent" / "templates" / "questions.yaml")
    assert set(templates) == {"logs", "events", "config", "state"}
    for specialist, entries in templates.items():
        assert entries, f"{specialist} has no templates"
        for t in entries:
            for slot in t.slots:
                assert "{" + slot + "}" in t.template, f"{t.id} declares unused slot {slot}"
```

The final test catches a whole class of bug at authoring time: a template whose declared slots and actual placeholders have drifted apart.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_templates.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'agent.questions'`

- [ ] **Step 3: Write `agent/templates/questions.yaml`**

Two templates per specialist is enough to measure whether choosing between them is reliable. Phase 3 expands the set.

```yaml
logs:
  - id: log_startup_failure
    template: "what did container {container} in pod {pod} in namespace {namespace} write before it last stopped?"
    slots: [container, pod, namespace]
  - id: log_pattern_search
    template: "does any pod in namespace {namespace} log a line matching {pattern}?"
    slots: [namespace, pattern]

events:
  - id: evt_for_object
    template: "what events has Kubernetes recorded for {kind}/{name} in namespace {namespace}?"
    slots: [kind, name, namespace]
  - id: evt_recent_warnings
    template: "what Warning events occurred in namespace {namespace} in the last {seconds} seconds?"
    slots: [namespace, seconds]

config:
  - id: cfg_missing_ref
    template: "does {kind}/{name} in namespace {namespace} reference a {resource_kind} named {resource_name} that is missing or lacks the key it needs?"
    slots: [kind, name, namespace, resource_kind, resource_name]
  - id: cfg_keys_present
    template: "which keys does {resource_kind} {name} in namespace {namespace} actually contain?"
    slots: [resource_kind, name, namespace]

state:
  - id: st_pod_condition
    template: "what is the current status and waiting reason of pod {pod} in namespace {namespace}?"
    slots: [pod, namespace]
  - id: st_pods_by_phase
    template: "which pods in namespace {namespace} are not in phase {phase}?"
    slots: [namespace, phase]
```

- [ ] **Step 4: Write `agent/questions.py`**

```python
from pathlib import Path

import yaml
from pydantic import BaseModel

PLACEHOLDER_VALUES = {"", "unknown", "todo", "none", "n/a", "null", "string", "value"}


class QuestionTemplate(BaseModel):
    id: str
    template: str
    slots: list[str]


class SlotFillDecision(BaseModel):
    specialist: str
    template_id: str
    slot_values: list[str]


def load_templates(path: Path) -> dict[str, list[QuestionTemplate]]:
    raw = yaml.safe_load(path.read_text())
    return {
        specialist: [QuestionTemplate(**t) for t in entries]
        for specialist, entries in raw.items()
    }


def _find(decision: SlotFillDecision, templates) -> QuestionTemplate | None:
    for t in templates.get(decision.specialist, []):
        if t.id == decision.template_id:
            return t
    return None


def validate_decision(
    decision: SlotFillDecision, templates: dict[str, list[QuestionTemplate]]
) -> list[str]:
    problems: list[str] = []

    if decision.specialist not in templates:
        known = ", ".join(sorted(templates))
        return [f"unknown specialist '{decision.specialist}'. Known specialists: {known}"]

    template = _find(decision, templates)
    if template is None:
        known = ", ".join(t.id for t in templates[decision.specialist])
        return [
            f"unknown template_id '{decision.template_id}' for specialist "
            f"'{decision.specialist}'. Known template ids: {known}"
        ]

    if len(decision.slot_values) != len(template.slots):
        problems.append(
            f"template '{template.id}' needs {len(template.slots)} slot values "
            f"but got {len(decision.slot_values)}"
        )
        return problems

    for slot, value in zip(template.slots, decision.slot_values, strict=True):
        stripped = value.strip()
        if stripped.lower() in PLACEHOLDER_VALUES:
            problems.append(f"slot '{slot}' has placeholder value {value!r}")
        elif stripped in (slot, "{" + slot + "}"):
            problems.append(f"slot '{slot}' was echoed back instead of filled")

    return problems


def render(decision: SlotFillDecision, templates: dict[str, list[QuestionTemplate]]) -> str:
    template = _find(decision, templates)
    if template is None:
        raise ValueError(f"cannot render unknown template '{decision.template_id}'")
    mapping = dict(zip(template.slots, decision.slot_values, strict=True))
    return template.template.format(**mapping)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `pytest tests/test_templates.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit the template machinery before running any model against it**

```bash
git add agent/questions.py agent/templates/questions.yaml tests/test_templates.py
git commit -m "feat: slot-fill question templates with validation"
```

- [ ] **Step 7: Write the spike script**

`spikes/slot_fill.py`:

```python
"""Phase 0 capability spike.

Measures whether the specialist-tier model can choose a question template and
fill its slots reliably enough for the supervisor design in the spec to hold.
Writes results/spike-slot-fill.json.
"""

import json

from agent.llm import RateLimiter, RequestCounter, generate_structured
from agent.models import load_routing, spec_for
from agent.questions import SlotFillDecision, load_templates, render, validate_decision
from config.settings import REPO_ROOT, settings

CASES = [
    (
        "the checkout service is returning 503s",
        "Pod checkout-8a2b in namespace shop is Pending with waiting reason "
        "CreateContainerConfigError. Deployment checkout reads DB_HOST from configmap app-config.",
        "config",
    ),
    (
        "the worker keeps dying",
        "Pod worker-4c1d in namespace jobs has restarted 7 times. Last state terminated, "
        "exit code 137. Container name is worker.",
        "logs",
    ),
    (
        "nothing is starting up in the batch namespace",
        "Three pods in namespace batch are Pending. No containerStatuses are populated yet.",
        "events",
    ),
    (
        "the api is up but unreachable",
        "Service api in namespace edge has no ready endpoints. Pod api-11f2 exists in namespace edge.",
        "state",
    ),
]

PROMPT = """You are choosing the next investigative question to ask a specialist.

Incident: {incident}

Evidence so far:
{evidence}

Available specialists and their question templates:
{catalogue}

Choose exactly one specialist and one template_id from that specialist's list.
Then provide slot_values as a flat list of strings, in the same order as the
template's slots, filled with concrete values taken from the evidence above.
Never leave a slot empty and never repeat the slot name as its value.
"""


def _catalogue(templates) -> str:
    lines = []
    for specialist, entries in templates.items():
        lines.append(f"{specialist}:")
        for t in entries:
            lines.append(f"  - template_id: {t.id}, slots in order: {t.slots}")
    return "\n".join(lines)


def main(trials_per_case: int = 5) -> None:
    templates = load_templates(REPO_ROOT / "agent" / "templates" / "questions.yaml")
    routing = load_routing(settings.model_routing_path)
    spec = spec_for("specialists", routing)
    limiter = RateLimiter(settings.requests_per_minute)
    counter = RequestCounter()

    records = []
    for incident, evidence, expected in CASES:
        prompt = PROMPT.format(
            incident=incident, evidence=evidence, catalogue=_catalogue(templates)
        )
        for trial in range(trials_per_case):
            record = {
                "incident": incident,
                "expected_specialist": expected,
                "trial": trial,
                "parse_failed": False,
                "problems": [],
                "specialist": None,
                "rendered": None,
            }
            try:
                decision = generate_structured(
                    prompt=prompt,
                    schema=SlotFillDecision,
                    role="specialists",
                    spec=spec,
                    limiter=limiter,
                    counter=counter,
                )
            except ValueError as exc:
                record["parse_failed"] = True
                record["problems"] = [str(exc)[:300]]
            else:
                problems = validate_decision(decision, templates)
                record["specialist"] = decision.specialist
                record["problems"] = problems
                if not problems:
                    record["rendered"] = render(decision, templates)
            records.append(record)

    total = len(records)
    parse_failures = sum(r["parse_failed"] for r in records)
    invalid = sum(bool(r["problems"]) for r in records)
    right_specialist = sum(r["specialist"] == r["expected_specialist"] for r in records)

    summary = {
        "model": spec.model,
        "trials": total,
        "requests_made": counter.total,
        "parse_failure_rate": round(parse_failures / total, 3),
        "invalid_decision_rate": round(invalid / total, 3),
        "usable_rate": round((total - invalid) / total, 3),
        "specialist_match_rate": round(right_specialist / total, 3),
    }

    settings.results_dir.mkdir(exist_ok=True)
    out = settings.results_dir / "spike-slot-fill.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
```

Twenty trials against Flash-Lite. At twelve requests per minute that is under two minutes and a negligible slice of the daily allowance.

Note that `specialist_match_rate` is softer evidence than the other numbers. The "expected" specialist is one reasonable choice rather than the only defensible one, so read it as a signal about whether routing is sensible, not as an accuracy score.

- [ ] **Step 8: Run the spike**

Run: `python -m spikes.slot_fill`
Expected: a JSON summary printed, and `results/spike-slot-fill.json` written.

- [ ] **Step 9: Record the result and decide**

Add to `DECISIONS.md` under a new dated heading, filling in the real numbers:

```markdown
## 2026-09-1X, Phase 0 spike result

- Slot-fill spike on <model>: usable_rate <X>, parse_failure_rate <Y>, specialist_match_rate <Z> over 20 trials. Raw data in `results/spike-slot-fill.json`.
- Decision: <one of the three below>.
```

Read the outcome against these thresholds:

| `usable_rate` | What it means | What to do |
|---|---|---|
| 0.90 or above | The design in section 5.4 holds | Proceed to Phase 1 unchanged |
| 0.70 to 0.90 | Workable with a retry | Add one constrained retry that feeds the validation problems back in, record it as a decision, proceed |
| below 0.70 | The specialist tier is too weak for slot filling | Move specialists up to the flash tier in `config/models.yaml`, rerun the spike, and if it still fails, revisit section 5.4 before building Phase 1 |

This is the phase's whole purpose. Do not proceed to Phase 1 without an entry in `DECISIONS.md` recording which branch was taken.

- [ ] **Step 10: Commit**

```bash
git add DECISIONS.md spikes/ results/spike-slot-fill.json
git commit -m "feat: slot-fill capability spike and Phase 0 result"
```

---

## Phase 0 acceptance

All four must hold before Phase 1 begins.

1. `pytest` passes with no integration tests deselected as failures, and `pytest -m integration` passes against a live cluster.
2. The kind cluster comes up and tears down repeatably, and creating it twice is a no-op.
3. Scenario 001 breaks and its `verify.sh` exits zero, proven by the integration test in Task 4.
4. `results/spike-slot-fill.json` exists, and `DECISIONS.md` records the number and which branch of the threshold table was taken.

---

## What Phase 1 picks up

The MCP server with at least eight read-only tools, flat schemas, and recoverable error messages. Four more scenarios spanning the difficulty tiers. A manual tool-runner script. The acceptance gate is that a human can diagnose every scenario from the tool output alone, because if a person cannot, no agent will.
