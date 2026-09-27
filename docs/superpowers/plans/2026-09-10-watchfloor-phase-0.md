# Watchfloor Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project foundations and answer the single riskiest open question in the design, which is whether a Flash-tier model can reliably choose a question template and fill its slots.

**Architecture:** Six tasks. The first four are pure Python and shell with no model calls and no cluster required for their unit tests: project skeleton and model routing, the scenario label contract plus the first seeded scenario, kind cluster management, and a scenario script runner. The fifth adds a rate-limited model client. The sixth runs the capability spike and records its result in `DECISIONS.md`. Every task ends with a commit.

> **Amended 2026-09-27.** A pre-flight scan before resuming Tasks 3 to 6 found defects in this plan's own text. Every code block in Tasks 3 to 6 was then executed in a scratch copy of the repo before being handed to an implementer. Each change and its reason is in `DECISIONS.md` under 2026-09-27.

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
- Harness code never reads or writes the user's `~/.kube/config`. Every `kind` and `kubectl` invocation is pinned to `settings.kubeconfig_path`.
- Test fixtures use placeholder model ids such as `test-model-a`, never a real one. The constraint above says anywhere, and that includes test code.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest configuration |
| `config/settings.py` | Environment-derived settings, one `Settings` class, including Watchfloor's private kubeconfig path |
| `config/models.yaml` | Model routing by node role. The only place model names appear |
| `agent/models.py` | Loads and validates model routing, resolves a role to a `ModelSpec` |
| `agent/llm.py` | Rate limiting, retry on 429 and 5xx, request counting, provider adapter |
| `agent/templates/questions.yaml` | Slot-fill question templates, keyed by specialist |
| `agent/questions.py` | Loads templates, validates a decision against them |
| `cluster/manage.py` | Idempotent kind cluster create, delete and existence check, every command pinned to the private kubeconfig |
| `scenarios/taxonomy.py` | The fixed fifteen root-cause categories |
| `scenarios/label.py` | `Label` schema and loader, validates against the taxonomy |
| `scenarios/runner.py` | Runs a scenario's break, verify and cleanup scripts in their own process group, pinned to the private kubeconfig |
| `scenarios/001-missing-configmap-key/` | First seeded scenario, five files |
| `spikes/slot_fill.py` | The Phase 0 capability spike, injectable and tested offline, writes JSON to `results/` |
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

## Task 3: Idempotent kind cluster management, pinned to a private kubeconfig

> **Amended 2026-09-27.** The first implementation (commit `e2c4ddd`) followed the original text of this task, and two defects in that text are corrected here. `list_clusters` now raises when `kind` fails instead of reading the failure as "no clusters exist". And every `kind` command is pinned to Watchfloor's own kubeconfig, so the harness can never switch or mutate the user's other clusters. The existing code is brought into line by a fix round rather than rebuilt. Reasons in `DECISIONS.md` under 2026-09-27.

**Files:**
- Modify: `cluster/manage.py`, `config/settings.py` (add `kubeconfig_path`), `.gitignore` (add `.watchfloor/`)
- Unchanged: `cluster/kind-config.yaml`
- Test: `tests/test_cluster_manage.py` (rewritten)

**Interfaces:**
- Consumes: `config.settings.settings.cluster_name`, `config.settings.REPO_ROOT`.
- Produces:
  - `config.settings.Settings.kubeconfig_path: Path`, defaulting to `REPO_ROOT / ".watchfloor" / "kubeconfig"`.
  - `cluster.manage.list_clusters(runner=...) -> list[str]`, raising `RuntimeError` that carries kind's stderr when `kind get clusters` exits non-zero.
  - `cluster.manage.cluster_exists(name, runner=...) -> bool`.
  - `cluster.manage.export_kubeconfig(name, *, kubeconfig: Path, runner=...) -> Path`.
  - `cluster.manage.create_cluster(name, *, kubeconfig: Path, config_path: Path | None = None, runner=...) -> bool`. True when it created the cluster. False when the cluster already existed, in which case it re-exports the kubeconfig so the private file is always present and current.
  - `cluster.manage.delete_cluster(name, *, kubeconfig: Path, runner=...) -> bool`.

`kubeconfig` is a required keyword argument with no default, deliberately. No code path in this module can fall back to `$KUBECONFIG` or `~/.kube/config`.

A `runner` parameter is injected so unit tests never shell out. Default is `subprocess.run`.

**Why the pinning matters.** Without `--kubeconfig`, `kind create cluster` rewrites `~/.kube/config` and switches the user's current context to `kind-watchfloor`, silently redirecting every `kubectl` command they run for other projects. Worse, on a later run where the cluster already exists, `create_cluster` switches nothing, so scenario scripts would act on whichever context the user last selected. The machine this plan runs on has `kind-miniargo` and `kind-nimbus` contexts, so this is a live risk.

- [ ] **Step 1: Rewrite the test file**

`tests/test_cluster_manage.py`:

```python
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cluster.manage import (
    cluster_exists,
    create_cluster,
    delete_cluster,
    export_kubeconfig,
    list_clusters,
)


@dataclass
class FakeRunner:
    """Stands in for subprocess.run.

    `kind get clusters` prints `clusters` and exits with `list_returncode`.
    Every other command exits with `returncode`. Both share `stderr`.
    """

    clusters: str = ""
    list_returncode: int = 0
    returncode: int = 0
    stderr: str = ""
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[:3] == ["kind", "get", "clusters"]:
            return subprocess.CompletedProcess(cmd, self.list_returncode, self.clusters, self.stderr)
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)

    def ran(self, *prefix: str) -> list[list[str]]:
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]


def _pinned_to(cmd: list[str], kubeconfig: Path) -> bool:
    return "--kubeconfig" in cmd and cmd[cmd.index("--kubeconfig") + 1] == str(kubeconfig)


def test_list_clusters_splits_lines():
    assert list_clusters(runner=FakeRunner(clusters="watchfloor\nother\n")) == ["watchfloor", "other"]


def test_list_clusters_handles_no_clusters():
    assert list_clusters(runner=FakeRunner(clusters="\n")) == []


def test_list_clusters_raises_with_kinds_stderr_when_kind_fails():
    runner = FakeRunner(list_returncode=1, stderr="Cannot connect to the Docker daemon")
    with pytest.raises(RuntimeError) as exc:
        list_clusters(runner=runner)
    assert "Cannot connect to the Docker daemon" in str(exc.value)


def test_cluster_exists_is_true_when_named():
    assert cluster_exists("watchfloor", runner=FakeRunner(clusters="watchfloor\n")) is True


def test_create_refreshes_the_kubeconfig_instead_of_creating_when_the_cluster_exists(tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    runner = FakeRunner(clusters="watchfloor\n")
    assert create_cluster("watchfloor", kubeconfig=kubeconfig, runner=runner) is False
    assert runner.ran("kind", "create") == []
    exports = runner.ran("kind", "export", "kubeconfig")
    assert len(exports) == 1
    assert _pinned_to(exports[0], kubeconfig)


def test_create_runs_kind_create_pinned_to_the_private_kubeconfig(tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    runner = FakeRunner(clusters="")
    assert create_cluster("watchfloor", kubeconfig=kubeconfig, runner=runner) is True
    (create,) = runner.ran("kind", "create", "cluster")
    assert create[create.index("--name") + 1] == "watchfloor"
    assert _pinned_to(create, kubeconfig)


def test_create_raises_with_kinds_stderr_when_create_fails(tmp_path):
    runner = FakeRunner(clusters="", returncode=1, stderr="node image pull failed")
    with pytest.raises(RuntimeError) as exc:
        create_cluster("watchfloor", kubeconfig=tmp_path / "kubeconfig", runner=runner)
    assert "node image pull failed" in str(exc.value)


def test_delete_is_a_noop_when_absent(tmp_path):
    runner = FakeRunner(clusters="")
    assert delete_cluster("watchfloor", kubeconfig=tmp_path / "kubeconfig", runner=runner) is False
    assert runner.ran("kind", "delete") == []


def test_delete_runs_kind_delete_pinned_to_the_private_kubeconfig(tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    runner = FakeRunner(clusters="watchfloor\n")
    assert delete_cluster("watchfloor", kubeconfig=kubeconfig, runner=runner) is True
    (delete,) = runner.ran("kind", "delete", "cluster")
    assert _pinned_to(delete, kubeconfig)


def test_delete_raises_with_kinds_stderr_when_delete_fails(tmp_path):
    runner = FakeRunner(clusters="watchfloor\n", returncode=1, stderr="container still running")
    with pytest.raises(RuntimeError) as exc:
        delete_cluster("watchfloor", kubeconfig=tmp_path / "kubeconfig", runner=runner)
    assert "container still running" in str(exc.value)


def test_delete_raises_instead_of_reporting_a_noop_when_listing_fails(tmp_path):
    runner = FakeRunner(list_returncode=1, stderr="Cannot connect to the Docker daemon")
    with pytest.raises(RuntimeError):
        delete_cluster("watchfloor", kubeconfig=tmp_path / "kubeconfig", runner=runner)
    assert runner.ran("kind", "delete") == []


def test_export_kubeconfig_creates_the_parent_directory(tmp_path):
    kubeconfig = tmp_path / "nested" / "dir" / "kubeconfig"
    runner = FakeRunner()
    assert export_kubeconfig("watchfloor", kubeconfig=kubeconfig, runner=runner) == kubeconfig
    assert kubeconfig.parent.is_dir()
    (export,) = runner.ran("kind", "export", "kubeconfig")
    assert _pinned_to(export, kubeconfig)


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.mark.integration
def test_create_is_idempotent_and_never_touches_the_users_kubeconfig():
    from config.settings import settings

    users_kubeconfig = Path.home() / ".kube" / "config"
    before = _digest(users_kubeconfig)

    create_cluster(settings.cluster_name, kubeconfig=settings.kubeconfig_path)
    assert create_cluster(settings.cluster_name, kubeconfig=settings.kubeconfig_path) is False
    assert cluster_exists(settings.cluster_name) is True
    assert settings.kubeconfig_path.exists()

    assert _digest(users_kubeconfig) == before, "~/.kube/config was modified"
```

`test_delete_raises_instead_of_reporting_a_noop_when_listing_fails` is the review finding expressed as a test: before this amendment, that call returned `False` and reported a clean no-op without having looked.

The integration test proves the isolation directly: it hashes `~/.kube/config` before and after, so any write to it at all, not just a context switch, fails the test.

- [ ] **Step 2: Run the tests and confirm they fail against the current code**

Run: `.venv/bin/pytest tests/test_cluster_manage.py -v`
Expected: FAIL. `ImportError` for `export_kubeconfig`, since the current module does not define it.

- [ ] **Step 3: Add the private kubeconfig path to settings, and ignore its directory**

In `config/settings.py`, add this field to `Settings` directly after `results_dir`:

```python
    kubeconfig_path: Path = REPO_ROOT / ".watchfloor" / "kubeconfig"
```

Append to `.gitignore`:

```
.watchfloor/
```

- [ ] **Step 4: Replace `cluster/manage.py`**

```python
import subprocess
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "kind-config.yaml"


def _run(cmd: list[str], runner=subprocess.run) -> subprocess.CompletedProcess:
    return runner(cmd, capture_output=True, text=True, check=False)


def _checked(cmd: list[str], runner=subprocess.run) -> subprocess.CompletedProcess:
    result = _run(cmd, runner=runner)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed with exit code {result.returncode}:\n"
            f"{result.stderr.strip()}"
        )
    return result


def list_clusters(runner=subprocess.run) -> list[str]:
    result = _checked(["kind", "get", "clusters"], runner=runner)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cluster_exists(name: str, runner=subprocess.run) -> bool:
    return name in list_clusters(runner=runner)


def export_kubeconfig(name: str, *, kubeconfig: Path, runner=subprocess.run) -> Path:
    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    _checked(
        ["kind", "export", "kubeconfig", "--name", name, "--kubeconfig", str(kubeconfig)],
        runner=runner,
    )
    return kubeconfig


def create_cluster(
    name: str,
    *,
    kubeconfig: Path,
    config_path: Path | None = None,
    runner=subprocess.run,
) -> bool:
    if cluster_exists(name, runner=runner):
        export_kubeconfig(name, kubeconfig=kubeconfig, runner=runner)
        return False
    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    config = config_path or DEFAULT_CONFIG
    _checked(
        [
            "kind", "create", "cluster",
            "--name", name,
            "--config", str(config),
            "--kubeconfig", str(kubeconfig),
        ],
        runner=runner,
    )
    return True


def delete_cluster(name: str, *, kubeconfig: Path, runner=subprocess.run) -> bool:
    if not cluster_exists(name, runner=runner):
        return False
    _checked(
        ["kind", "delete", "cluster", "--name", name, "--kubeconfig", str(kubeconfig)],
        runner=runner,
    )
    return True
```

Every `kind` call now goes through `_checked`, so the listing call fails as loudly as create and delete, and they all report kind's stderr.

- [ ] **Step 5: Run the unit tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_cluster_manage.py -v`
Expected: 12 passed, 1 deselected.

Then the whole default suite, to confirm the settings change broke nothing: `.venv/bin/pytest -v`.

- [ ] **Step 6: Run the integration test against a real cluster**

Docker must be running. The first run pulls kind's node image and can take several minutes, so allow up to ten.

Run: `.venv/bin/pytest tests/test_cluster_manage.py -v -m integration`
Expected: PASS.

Then confirm the cluster is reachable only through the private kubeconfig:

```bash
kubectl --kubeconfig .watchfloor/kubeconfig get nodes
kubectl config current-context
```

Expected: three `Ready` nodes from the first command. The second prints whatever the user's context was before, not `kind-watchfloor`.

- [ ] **Step 7: Commit**

```bash
git add cluster/manage.py config/settings.py .gitignore tests/test_cluster_manage.py
git commit -m "fix: fail loudly when kind fails, pin every kind call to a private kubeconfig"
```

---

## Task 4: Scenario script runner, and making scenario 001 match its label

> **Amended 2026-09-27.** Three changes from the original text. The runner pins every script to Watchfloor's private kubeconfig, for the reason given in Task 3. Scripts run in their own process group and the whole group is killed on timeout, so a hung `kubectl` cannot outlive its script and interfere with the next scenario. And scenario 001's fixture is corrected so the cluster actually reaches the state its pre-registered label describes. The label itself is not touched. Reasons in `DECISIONS.md` under 2026-09-27.

**Files:**
- Create: `scenarios/runner.py`
- Modify: `scenarios/001-missing-configmap-key/manifest.yaml`, `break.sh`, `verify.sh`
- Must not modify: `scenarios/001-missing-configmap-key/label.yaml`
- Test: `tests/test_scenario_runner.py`

**Interfaces:**
- Consumes: `config.settings.settings.kubeconfig_path`, `config.settings.settings.scenarios_dir`, and `cluster.manage.create_cluster(name, *, kubeconfig)` in the integration test only.
- Produces: `scenarios.runner.ScriptResult` with fields `script: str`, `exit_code: int`, `stdout: str`, `stderr: str`, `duration_s: float`; `scenarios.runner.run_script(scenario_dir: Path, script: str, *, kubeconfig: Path, timeout: int = 180) -> ScriptResult`; and `break_scenario`, `verify_scenario`, `cleanup_scenario`, each `(scenario_dir: Path, *, kubeconfig: Path) -> ScriptResult`.

As in Task 3, `kubeconfig` is required with no default.

- [ ] **Step 1: Write the failing test**

`tests/test_scenario_runner.py`:

```python
import os
import subprocess
import time
from pathlib import Path

import pytest

from scenarios.runner import cleanup_scenario, run_script, verify_scenario


def _write_script(d: Path, name: str, body: str) -> None:
    path = d / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


@pytest.fixture
def kubeconfig(tmp_path) -> Path:
    return tmp_path / "kubeconfig"


def test_captures_stdout_and_zero_exit(tmp_path, kubeconfig):
    _write_script(tmp_path, "break.sh", "echo broke it")
    result = run_script(tmp_path, "break.sh", kubeconfig=kubeconfig)
    assert result.exit_code == 0
    assert "broke it" in result.stdout
    assert result.script == "break.sh"
    assert result.duration_s >= 0


def test_captures_nonzero_exit_and_stderr(tmp_path, kubeconfig):
    _write_script(tmp_path, "verify.sh", "echo nope >&2; exit 1")
    result = verify_scenario(tmp_path, kubeconfig=kubeconfig)
    assert result.exit_code == 1
    assert "nope" in result.stderr


def test_scripts_see_only_the_private_kubeconfig(tmp_path, kubeconfig):
    _write_script(tmp_path, "cleanup.sh", 'echo "KUBECONFIG=$KUBECONFIG"')
    result = cleanup_scenario(tmp_path, kubeconfig=kubeconfig)
    assert result.stdout.strip() == f"KUBECONFIG={kubeconfig}"


def test_scripts_run_inside_their_scenario_directory(tmp_path, kubeconfig):
    _write_script(tmp_path, "break.sh", "pwd -P")
    result = run_script(tmp_path, "break.sh", kubeconfig=kubeconfig)
    assert result.stdout.strip() == str(tmp_path.resolve())


def test_missing_script_names_what_is_present(tmp_path, kubeconfig):
    _write_script(tmp_path, "break.sh", "true")
    with pytest.raises(FileNotFoundError) as exc:
        run_script(tmp_path, "verify.sh", kubeconfig=kubeconfig)
    assert "verify.sh" in str(exc.value)
    assert "break.sh" in str(exc.value)


def test_timeout_kills_the_script_and_everything_it_started(tmp_path, kubeconfig):
    pidfile = tmp_path / "child.pid"
    _write_script(tmp_path, "break.sh", f'sleep 30 & echo $! > "{pidfile}"; wait')
    with pytest.raises(subprocess.TimeoutExpired):
        run_script(tmp_path, "break.sh", kubeconfig=kubeconfig, timeout=1)

    child = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"backgrounded child {child} outlived the timed-out script")
```

`test_scripts_see_only_the_private_kubeconfig` pins the safety property: `kubectl` treats `KUBECONFIG` as the complete list of config files, so a script that sees only this path cannot reach any other cluster.

`test_timeout_kills_the_script_and_everything_it_started` backgrounds a child and records its pid. Under a plain `subprocess.run` timeout only the script dies and the child lives on for thirty seconds. That is the failure mode this test exists to rule out.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_scenario_runner.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scenarios.runner'`

- [ ] **Step 3: Write `scenarios/runner.py`**

```python
import contextlib
import os
import signal
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


def run_script(
    scenario_dir: Path, script: str, *, kubeconfig: Path, timeout: int = 180
) -> ScriptResult:
    """Run one scenario script, pinned to Watchfloor's private kubeconfig.

    The script runs with `scenario_dir` as its working directory and in its own
    process group. On timeout the whole group is killed, so a hung kubectl or a
    backgrounded child cannot outlive the script and act on the next scenario.
    """
    path = scenario_dir / script
    if not path.exists():
        present = ", ".join(sorted(p.name for p in scenario_dir.glob("*.sh"))) or "none"
        raise FileNotFoundError(
            f"'{script}' not found in {scenario_dir}. Scripts present: {present}"
        )

    started = time.monotonic()
    proc = subprocess.Popen(
        [str(path)],
        cwd=scenario_dir,
        env={**os.environ, "KUBECONFIG": str(kubeconfig)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise

    return ScriptResult(
        script=script,
        exit_code=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=time.monotonic() - started,
    )


def break_scenario(scenario_dir: Path, *, kubeconfig: Path) -> ScriptResult:
    return run_script(scenario_dir, "break.sh", kubeconfig=kubeconfig)


def verify_scenario(scenario_dir: Path, *, kubeconfig: Path) -> ScriptResult:
    return run_script(scenario_dir, "verify.sh", kubeconfig=kubeconfig)


def cleanup_scenario(scenario_dir: Path, *, kubeconfig: Path) -> ScriptResult:
    return run_script(scenario_dir, "cleanup.sh", kubeconfig=kubeconfig)
```

`start_new_session=True` makes the script the leader of a new process group, so `os.killpg` reaches everything it started. `ProcessLookupError` is suppressed for the race where the group finishes exiting between the timeout and the kill.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_scenario_runner.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit the runner**

```bash
git add scenarios/runner.py tests/test_scenario_runner.py
git commit -m "feat: scenario script runner, process-group timeouts, pinned kubeconfig"
```

- [ ] **Step 6: Correct scenario 001 so the cluster reaches the state its label describes**

The pre-registered label says the checkout container cannot be created, **so the Service has no ready endpoints**. The original `break.sh` did not produce that second half. With one replica, a Deployment's rolling update has `maxSurge` 1 and `maxUnavailable` 0, so `rollout restart` creates the new pod first and keeps the old one until the new one is Ready. The new one never becomes Ready, so the old pod stays Ready and the Service keeps an endpoint. The incident text, "returning 503s", was therefore untrue of the cluster the scenario built.

The fix brings the fixture into line with the label. **Do not edit `label.yaml`.** It was committed before any agent existed, and that ordering is the credibility argument for the whole evaluation.

Three edits.

In `manifest.yaml`, replace the `checkout` container definition so the healthy baseline actually serves on the port its Service targets. The original container slept and listened on nothing, so the baseline was not genuinely healthy either:

```yaml
        - name: checkout
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "starting against $DB_HOST"
              mkdir -p /www && echo ok > /www/index.html
              exec httpd -f -p 8080 -h /www
          ports:
            - containerPort: 8080
          env:
            - name: DB_HOST
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: DB_HOST
```

Replace `break.sh` entirely:

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl apply -f "$DIR/manifest.yaml"
kubectl -n shop rollout status deployment/checkout --timeout=120s

kubectl -n shop patch configmap app-config \
  --type=json -p='[{"op": "remove", "path": "/data/DB_HOST"}]'

# Delete the running pod rather than using `rollout restart`. With one replica
# a rolling restart keeps the old pod Ready until its replacement is Ready,
# which never happens here, so the Service would keep serving. Deleting the pod
# makes the ReplicaSet create a replacement that cannot start, and leaves
# nothing Ready behind the Service.
kubectl -n shop delete pod -l app=checkout --wait=false
```

Replace `verify.sh` entirely, so it checks both halves of the label's claim rather than only the first:

```bash
#!/usr/bin/env bash
set -uo pipefail

# The fault is live when both halves of the label's claim hold at once: a
# checkout pod is stuck in CreateContainerConfigError, and no checkout pod is
# Ready, so the Service has no endpoints to send traffic to.
for _ in $(seq 1 45); do
  reasons=$(kubectl -n shop get pods -l app=checkout \
    -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null)
  ready=$(kubectl -n shop get pods -l app=checkout \
    -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
  if echo "$reasons" | grep -q CreateContainerConfigError && ! echo "$ready" | grep -q True; then
    echo "fault confirmed: checkout in CreateContainerConfigError with no Ready pods"
    exit 0
  fi
  sleep 2
done

echo "fault NOT confirmed after 90s. Waiting reasons: ${reasons:-none}. Ready statuses: ${ready:-none}" >&2
exit 1
```

Confirm all three scripts are still executable afterwards: `git ls-files -s scenarios/001-missing-configmap-key/` must show `100755` for each `.sh`.

- [ ] **Step 7: Add the end-to-end integration test**

Append to `tests/test_scenario_runner.py`:

```python
@pytest.mark.integration
def test_scenario_001_breaks_and_verifies():
    from cluster.manage import create_cluster
    from config.settings import settings
    from scenarios.runner import break_scenario

    kubeconfig = settings.kubeconfig_path
    create_cluster(settings.cluster_name, kubeconfig=kubeconfig)
    scenario = settings.scenarios_dir / "001-missing-configmap-key"
    try:
        broke = break_scenario(scenario, kubeconfig=kubeconfig)
        assert broke.exit_code == 0, broke.stdout + broke.stderr
        verified = verify_scenario(scenario, kubeconfig=kubeconfig)
        assert verified.exit_code == 0, verified.stdout + verified.stderr
    finally:
        cleanup_scenario(scenario, kubeconfig=kubeconfig)
```

The assertion messages carry the scripts' own output, so a failure says what the cluster was doing instead of `assert 1 == 0`.

- [ ] **Step 8: Run it and confirm the fault is real**

Docker must be running. The first run pulls `busybox:1.36` into the cluster.

Run: `.venv/bin/pytest tests/test_scenario_runner.py -v -m integration`
Expected: PASS. This is the Phase 0 acceptance gate for the scenario contract.

- [ ] **Step 9: Prove the label was not touched**

Run: `git diff --stat 4befc2e -- scenarios/001-missing-configmap-key/label.yaml`
Expected: no output.

- [ ] **Step 10: Commit the scenario correction**

```bash
git add scenarios/001-missing-configmap-key/manifest.yaml \
        scenarios/001-missing-configmap-key/break.sh \
        scenarios/001-missing-configmap-key/verify.sh \
        tests/test_scenario_runner.py
git commit -m "fix: scenario 001 now produces the no-ready-endpoints state its label claims"
```

---

## Task 5: Rate-limited, retrying, counted model client

> **Amended 2026-09-27.** Four changes from the original text, each verified against the installed `google-genai` 2.22.0 rather than assumed. The schema is sent through `response_json_schema`, because 2.22.0 documents `response_schema` as an OpenAPI subset and `response_json_schema` as the field that takes the JSON Schema `model_json_schema()` produces. Transient failures, 429 and 5xx, are retried with exponential backoff, which the design required and the original text omitted. Every attempt is counted and rate-limited, including retries, and the SDK's own retry is off by default, so nothing is retried twice or retried invisibly. And test fixtures use a placeholder model id, per the global constraint. Reasons in `DECISIONS.md` under 2026-09-27.

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `agent.models.ModelSpec`, `config.settings.settings`.
- Produces:
  - `agent.llm.RetryableError(status_code: int, message: str)`, with attribute `status_code`.
  - `agent.llm.RateLimiter(rpm: int, clock=..., sleep=...)` with `acquire() -> None`, raising `ValueError` for a non-positive `rpm`.
  - `agent.llm.RequestCounter` with `total: int`, `by_role: dict[str, int]`, `record(role: str) -> None`.
  - `agent.llm.generate_structured(prompt: str, schema: type[BaseModel], role: str, *, spec: ModelSpec, limiter: RateLimiter, counter: RequestCounter, call=..., max_attempts: int = 5, backoff_base_s: float = 4.0, sleep=time.sleep) -> BaseModel`.

The `call` parameter is the provider adapter. It must return the model's raw text, and must raise `RetryableError` for failures worth retrying. Injecting it is what lets every test in this task run offline, including the tests of the real Google adapter, which swap out only the SDK client underneath it.

**Retry arithmetic.** With the defaults, the waits between attempts are 4, 8, 16 and 32 seconds, so a persistent failure gives up after about a minute. That covers a per-minute quota window. It does not cover a daily cap, and nothing should: once the daily allowance is gone, failing after a minute is the right answer.

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:

```python
import pytest
from pydantic import BaseModel

from agent import llm
from agent.llm import RateLimiter, RequestCounter, RetryableError, generate_structured
from agent.models import ModelSpec

SPEC = ModelSpec(provider="google", model="test-model-a")


class Answer(BaseModel):
    verdict: str


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _free_limiter() -> RateLimiter:
    return RateLimiter(rpm=6000, clock=FakeClock(), sleep=lambda _: None)


def _generate(call, **overrides):
    kwargs = dict(
        prompt="anything",
        schema=Answer,
        role="supervisor",
        spec=SPEC,
        limiter=_free_limiter(),
        counter=RequestCounter(),
        call=call,
        sleep=lambda _: None,
    )
    kwargs.update(overrides)
    return generate_structured(**kwargs)


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


def test_limiter_rejects_a_non_positive_rate():
    with pytest.raises(ValueError):
        RateLimiter(rpm=0)


def test_counter_tracks_total_and_role():
    counter = RequestCounter()
    counter.record("supervisor")
    counter.record("supervisor")
    counter.record("logs")
    assert counter.total == 3
    assert counter.by_role == {"supervisor": 2, "logs": 1}


def test_generate_structured_returns_the_parsed_schema_and_counts():
    counter = RequestCounter()
    result = _generate(lambda **kw: '{"verdict": "ok"}', counter=counter)
    assert result.verdict == "ok"
    assert counter.total == 1


def test_generate_structured_raises_on_unparseable_output():
    with pytest.raises(ValueError) as exc:
        _generate(lambda **kw: "I think the answer is probably fine")
    assert "supervisor" in str(exc.value)


@pytest.mark.parametrize("raw", [None, ""])
def test_missing_or_empty_output_is_unparseable(raw):
    with pytest.raises(ValueError):
        _generate(lambda **kw: raw)


def test_retryable_errors_back_off_exponentially_and_every_attempt_counts():
    outcomes = iter(
        [RetryableError(429, "quota"), RetryableError(503, "overloaded"), '{"verdict": "ok"}']
    )

    def call(**kw):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    slept, counter = [], RequestCounter()
    result = _generate(call, counter=counter, sleep=slept.append)
    assert result.verdict == "ok"
    assert slept == [4.0, 8.0]
    assert counter.total == 3


def test_gives_up_after_max_attempts():
    def call(**kw):
        raise RetryableError(429, "quota")

    slept, counter = [], RequestCounter()
    with pytest.raises(RetryableError):
        _generate(call, counter=counter, sleep=slept.append, max_attempts=3)
    assert counter.total == 3
    assert slept == [4.0, 8.0]


def test_non_retryable_errors_fail_immediately():
    def call(**kw):
        raise RuntimeError("bad request")

    counter = RequestCounter()
    with pytest.raises(RuntimeError):
        _generate(call, counter=counter)
    assert counter.total == 1


def test_max_attempts_must_be_at_least_one():
    with pytest.raises(ValueError):
        _generate(lambda **kw: '{"verdict": "ok"}', max_attempts=0)


class _FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return type("Response", (), {"text": self.outcome})()


class _FakeClient:
    def __init__(self, outcome):
        self.models = _FakeModels(outcome)


@pytest.fixture
def with_key(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "google_api_key", "test-key")


def test_google_call_sends_the_schema_as_json_schema(monkeypatch, with_key):
    client = _FakeClient('{"verdict": "ok"}')
    monkeypatch.setattr(llm, "_google_client", lambda api_key: client)

    assert llm._google_call(prompt="p", spec=SPEC, schema=Answer) == '{"verdict": "ok"}'

    sent = client.models.kwargs
    assert sent["model"] == "test-model-a"
    assert sent["config"]["response_mime_type"] == "application/json"
    assert sent["config"]["response_json_schema"] == Answer.model_json_schema()
    assert "response_schema" not in sent["config"]


@pytest.mark.parametrize("code", [429, 500, 503])
def test_google_call_maps_transient_api_errors_to_retryable(monkeypatch, with_key, code):
    from google.genai import errors

    error_cls = errors.ClientError if code < 500 else errors.ServerError
    body = {"error": {"code": code, "message": "transient", "status": "UNAVAILABLE"}}
    client = _FakeClient(error_cls(code, body))
    monkeypatch.setattr(llm, "_google_client", lambda api_key: client)

    with pytest.raises(RetryableError) as exc:
        llm._google_call(prompt="p", spec=SPEC, schema=Answer)
    assert exc.value.status_code == code


def test_google_call_lets_non_transient_api_errors_through(monkeypatch, with_key):
    from google.genai import errors

    body = {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}
    client = _FakeClient(errors.ClientError(400, body))
    monkeypatch.setattr(llm, "_google_client", lambda api_key: client)

    with pytest.raises(errors.ClientError):
        llm._google_call(prompt="p", spec=SPEC, schema=Answer)


def test_google_call_without_a_key_says_where_to_put_it(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "google_api_key", "")
    with pytest.raises(RuntimeError) as exc:
        llm._google_call(prompt="p", spec=SPEC, schema=Answer)
    assert "WATCHFLOOR_GOOGLE_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)
```

The four `test_google_call_*` tests exercise the real adapter with only the SDK client replaced. They pin the three decisions most likely to drift as the SDK changes: which config field carries the schema, which status codes count as transient, and what happens without a key.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Write `agent/llm.py`**

```python
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from agent.models import ModelSpec

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RetryableError(Exception):
    """A provider failure worth retrying: rate limiting or a transient server error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code


class RateLimiter:
    """Spaces calls to at most `rpm` per minute."""

    def __init__(self, rpm: int, clock=time.monotonic, sleep=time.sleep):
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
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


@lru_cache(maxsize=1)
def _google_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def _google_call(*, prompt: str, spec: ModelSpec, schema: type[BaseModel]) -> str:
    from google.genai import errors

    from config.settings import settings

    if not settings.google_api_key:
        raise RuntimeError(
            "WATCHFLOOR_GOOGLE_API_KEY is not set. Put it in .env at the repo root."
        )
    try:
        response = _google_client(settings.google_api_key).models.generate_content(
            model=spec.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),
            },
        )
    except errors.APIError as exc:
        if exc.code in RETRYABLE_STATUS_CODES:
            raise RetryableError(exc.code, str(exc)) from exc
        raise
    return response.text or ""


def generate_structured(
    prompt: str,
    schema: type[BaseModel],
    role: str,
    *,
    spec: ModelSpec,
    limiter: RateLimiter,
    counter: RequestCounter,
    call=_google_call,
    max_attempts: int = 5,
    backoff_base_s: float = 4.0,
    sleep=time.sleep,
) -> BaseModel:
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    for attempt in range(1, max_attempts + 1):
        limiter.acquire()
        counter.record(role)
        try:
            raw = call(prompt=prompt, spec=spec, schema=schema) or ""
            break
        except RetryableError:
            if attempt == max_attempts:
                raise
            sleep(backoff_base_s * 2 ** (attempt - 1))

    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Model output for role '{role}' did not match {schema.__name__}. "
            f"Raw output: {raw[:400]!r}"
        ) from exc
```

Unparseable output still raises `ValueError` and is never retried. The design routes it to a deterministic fallback in Phase 3, and retrying it here would hide the failure rate the Phase 0 spike exists to measure.

`by_role` uses a `defaultdict`, which compares equal to a plain dict, so the counter test holds.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: 18 passed. That is 15 test functions, two of them parametrized, one over two values and one over three.

- [ ] **Step 5: Verify the provider SDK against a live call, when a key is available**

Skip this step if `WATCHFLOOR_GOOGLE_API_KEY` is not set, and record it in the report as deferred. With a key in `.env`:

```bash
.venv/bin/python -c "
from agent.llm import RateLimiter, RequestCounter, generate_structured
from agent.models import load_routing, spec_for
from config.settings import settings
from pydantic import BaseModel

class Ping(BaseModel):
    answer: str

routing = load_routing(settings.model_routing_path)
counter = RequestCounter()
print(generate_structured(
    prompt='Reply with the JSON object {\"answer\": \"pong\"} and nothing else.',
    schema=Ping, role='supervisor',
    spec=spec_for('supervisor', routing),
    limiter=RateLimiter(settings.requests_per_minute),
    counter=counter,
), counter.total)
"
```

Expected: `answer='pong' 1`. If the SDK surface has moved again, fix `_google_call` only, and record the change in `DECISIONS.md`.

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: model client with rate limiting, retry on 429/5xx, and request counting"
```

---

## Task 6: The slot-fill capability spike

This is the reason Phase 0 exists. Everything above is scaffolding for this measurement.

> **Amended 2026-09-27.** Five changes from the original text. `SlotFillDecision.specialist` is a `Literal` of the four specialists, per the global constraint that every control-flow output is schema-constrained. A `QuestionTemplate` now refuses to load unless its declared slots and its placeholders match in both directions; the original test checked one direction only. The spike counts a trial as usable only if every identifier slot names something the model was actually shown, since a question about an invented pod is not usable. The prompt's catalogue shows each template's question text, not only its id, so the spike measures the task the real supervisor will face. And the spike is injectable, tested offline, and keeps partial results if the provider gives up mid-run. Reasons in `DECISIONS.md` under 2026-09-27.

**Files:**
- Create: `agent/templates/questions.yaml`, `agent/questions.py`
- Create: `spikes/__init__.py`, `spikes/slot_fill.py`
- Test: `tests/test_templates.py`, `tests/test_spike.py`

**Interfaces:**
- Consumes: `agent.llm.generate_structured`, `agent.llm.RateLimiter`, `agent.llm.RequestCounter`, `agent.llm.RetryableError`, `agent.models.load_routing`, `agent.models.spec_for`, `config.settings.settings`, `config.settings.REPO_ROOT`.
- Produces:
  - `agent.questions.Specialist`, the `Literal["logs", "events", "config", "state"]` type.
  - `agent.questions.QuestionTemplate` with fields `id: str`, `template: str`, `slots: list[str]`, raising `ValidationError` on construction when slots and placeholders disagree.
  - `agent.questions.SlotFillDecision` with fields `specialist: Specialist`, `template_id: str`, `slot_values: list[str]`.
  - `agent.questions.load_templates(path: Path) -> dict[str, list[QuestionTemplate]]`.
  - `agent.questions.validate_decision(decision, templates) -> list[str]`, empty when valid.
  - `agent.questions.slot_mapping(decision, templates) -> dict[str, str]`.
  - `agent.questions.render(decision, templates) -> str`.
  - `spikes.slot_fill.ungrounded_identifiers(mapping: dict[str, str], source_text: str) -> list[str]`.
  - `spikes.slot_fill.main(trials_per_case: int = 5, *, call=None, limiter=None, sleep=time.sleep, results_dir=None) -> dict`, returning the summary.

`slot_values` is a flat list positionally matching the template's declared slots. A nested object would be a harder ask of a small model, and the whole point of this task is to avoid that.

- [ ] **Step 1: Write the failing template test**

`tests/test_templates.py`:

```python
import pytest
from pydantic import ValidationError

from agent.questions import (
    QuestionTemplate,
    SlotFillDecision,
    load_templates,
    render,
    slot_mapping,
    validate_decision,
)
from config.settings import REPO_ROOT

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


def test_specialist_with_no_loaded_templates_is_a_problem():
    problems = validate_decision(_decision(specialist="logs"), TEMPLATES)
    assert any("logs" in p for p in problems)


def test_specialist_outside_the_four_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        _decision(specialist="network")


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


def test_slot_mapping_pairs_each_slot_with_its_value():
    assert slot_mapping(_decision(), TEMPLATES) == {
        "kind": "Deployment",
        "name": "checkout",
        "namespace": "shop",
        "resource_kind": "ConfigMap",
    }


def test_render_substitutes_positionally():
    assert render(_decision(), TEMPLATES) == (
        "does Deployment/checkout in namespace shop reference a missing ConfigMap?"
    )


def test_template_using_an_undeclared_placeholder_is_rejected():
    with pytest.raises(ValidationError) as exc:
        QuestionTemplate(id="drifted", template="is {pod} in {namespace}?", slots=["pod"])
    assert "drifted" in str(exc.value)


def test_template_declaring_a_slot_it_never_uses_is_rejected():
    with pytest.raises(ValidationError):
        QuestionTemplate(id="drifted", template="is {pod} healthy?", slots=["pod", "namespace"])


def test_shipped_templates_load_and_cover_all_four_specialists():
    templates = load_templates(REPO_ROOT / "agent" / "templates" / "questions.yaml")
    assert set(templates) == {"logs", "events", "config", "state"}
    for specialist, entries in templates.items():
        assert entries, f"{specialist} has no templates"
```

Loading the shipped templates is now itself the consistency check, because a drifted template raises during construction. The two `drifted` tests prove each direction is enforced.

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/pytest tests/test_templates.py -v`
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
from string import Formatter
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

Specialist = Literal["logs", "events", "config", "state"]

PLACEHOLDER_VALUES = {"", "unknown", "todo", "none", "n/a", "null", "string", "value"}


class QuestionTemplate(BaseModel):
    id: str
    template: str
    slots: list[str]

    @model_validator(mode="after")
    def slots_match_placeholders(self) -> "QuestionTemplate":
        placeholders = [name for _, name, _, _ in Formatter().parse(self.template) if name]
        if sorted(placeholders) != sorted(self.slots):
            raise ValueError(
                f"template '{self.id}' declares slots {self.slots} "
                f"but its text uses placeholders {placeholders}"
            )
        return self


class SlotFillDecision(BaseModel):
    specialist: Specialist
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


def slot_mapping(
    decision: SlotFillDecision, templates: dict[str, list[QuestionTemplate]]
) -> dict[str, str]:
    template = _find(decision, templates)
    if template is None:
        raise ValueError(
            f"unknown template '{decision.template_id}' for specialist '{decision.specialist}'"
        )
    return dict(zip(template.slots, decision.slot_values, strict=True))


def render(decision: SlotFillDecision, templates: dict[str, list[QuestionTemplate]]) -> str:
    mapping = slot_mapping(decision, templates)
    return _find(decision, templates).template.format(**mapping)
```

`Formatter().parse` is the standard library's own parser for `str.format` strings, so the validator sees exactly the placeholders `render` will substitute. Comparing sorted lists rather than sets also rejects a template that repeats a placeholder, which a positional `slot_values` list cannot express.

- [ ] **Step 5: Run the template tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_templates.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit the template machinery before any model runs against it**

```bash
git add agent/questions.py agent/templates/questions.yaml tests/test_templates.py
git commit -m "feat: slot-fill question templates with two-way slot validation"
```

- [ ] **Step 7: Write the failing spike test**

`tests/test_spike.py`:

```python
import json

from agent.llm import RateLimiter, RetryableError
from spikes import slot_fill


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _limiter() -> RateLimiter:
    return RateLimiter(rpm=6000, clock=FakeClock(), sleep=lambda _: None)


def _scripted(*responses):
    """A fake model that answers each call with the next scripted response."""
    queue = list(responses)

    def call(**kwargs):
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return call


GROUNDED = json.dumps({
    "specialist": "config",
    "template_id": "cfg_missing_ref",
    "slot_values": ["Deployment", "checkout", "shop", "ConfigMap", "app-config"],
})
INVENTED_POD = json.dumps({
    "specialist": "logs",
    "template_id": "log_startup_failure",
    "slot_values": ["worker", "worker-9zzz", "jobs"],
})
UNKNOWN_TEMPLATE = json.dumps({
    "specialist": "events",
    "template_id": "evt_nope",
    "slot_values": ["batch"],
})
UNPARSEABLE = "I would ask the state specialist about the api pod"


def test_identifier_grounding_matches_whole_tokens_only():
    text = "Pod checkout-8a2b in namespace shop"
    assert slot_fill.ungrounded_identifiers({"pod": "checkout-8a2b", "namespace": "shop"}, text) == []
    assert slot_fill.ungrounded_identifiers({"pod": "checkout", "namespace": "sho"}, text) == [
        "pod",
        "namespace",
    ]
    assert slot_fill.ungrounded_identifiers({"phase": "Running", "seconds": "600"}, text) == []


def test_spike_scores_a_mix_of_outcomes(tmp_path):
    summary = slot_fill.main(
        trials_per_case=1,
        call=_scripted(GROUNDED, INVENTED_POD, UNKNOWN_TEMPLATE, UNPARSEABLE),
        limiter=_limiter(),
        sleep=lambda _: None,
        results_dir=tmp_path,
    )

    assert summary["complete"] is True
    assert summary["trials"] == 4
    assert summary["requests_made"] == 4
    assert summary["parse_failure_rate"] == 0.25
    assert summary["invalid_decision_rate"] == 0.5
    assert summary["ungrounded_identifier_rate"] == 0.25
    assert summary["usable_rate"] == 0.25
    assert summary["specialist_match_rate"] == 0.75

    written = json.loads((tmp_path / "spike-slot-fill.json").read_text())
    assert written["summary"] == summary
    assert [r["usable"] for r in written["records"]] == [True, False, False, False]
    assert written["records"][1]["ungrounded_identifiers"] == ["pod"]


def test_spike_stops_cleanly_and_keeps_partial_results_when_the_provider_gives_up(tmp_path):
    exhausted = [RetryableError(429, "quota exhausted")] * 5
    summary = slot_fill.main(
        trials_per_case=1,
        call=_scripted(GROUNDED, *exhausted),
        limiter=_limiter(),
        sleep=lambda _: None,
        results_dir=tmp_path,
    )

    assert summary["complete"] is False
    assert "429" in summary["aborted_reason"]
    assert summary["trials"] == 1
    assert summary["requests_made"] == 6
    written = json.loads((tmp_path / "spike-slot-fill.json").read_text())
    assert len(written["records"]) == 1
```

The four scripted responses line up with the spike's four cases in order, one trial each. Case 2 is the one that matters most: `worker-9zzz` parses, passes validation and renders a sensible-looking question, but the evidence names `worker-4c1d`. Before this amendment it counted as usable.

In the second test, one good answer is followed by five `RetryableError`s, which is exactly `generate_structured`'s default attempt budget, so the run stops after 1 + 5 = 6 requests with one completed trial.

- [ ] **Step 8: Run it and confirm it fails**

Run: `.venv/bin/pytest tests/test_spike.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'spikes'`

- [ ] **Step 9: Write the spike**

Create an empty `spikes/__init__.py`, then `spikes/slot_fill.py`:

```python
"""Phase 0 capability spike.

Measures whether the specialist-tier model can choose a question template and
fill its slots reliably enough for the supervisor design in the spec to hold.
Writes results/spike-slot-fill.json.

A trial counts as usable only when the output parses, passes template
validation, and every identifier slot names something that actually appears in
the incident or evidence the model was shown. A question about a pod the model
invented would send a specialist after something that does not exist.
"""

import json
import re
import time
from pathlib import Path

from agent.llm import RateLimiter, RequestCounter, RetryableError, generate_structured
from agent.models import load_routing, spec_for
from agent.questions import (
    SlotFillDecision,
    load_templates,
    render,
    slot_mapping,
    validate_decision,
)
from config.settings import REPO_ROOT, settings

IDENTIFIER_SLOTS = frozenset({"namespace", "pod", "container", "name", "resource_name"})

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
            lines.append(
                f'  - template_id: {t.id}, asks: "{t.template}", slots in order: {t.slots}'
            )
    return "\n".join(lines)


def ungrounded_identifiers(mapping: dict[str, str], source_text: str) -> list[str]:
    """Identifier slots whose value never appears as a whole token in `source_text`.

    Hyphens count as part of a token, so "checkout" does not match inside
    "checkout-8a2b". Kubernetes names are hyphenated, and a bare prefix names a
    different object.
    """
    ungrounded = []
    for slot, value in mapping.items():
        if slot not in IDENTIFIER_SLOTS:
            continue
        token = re.escape(value.strip())
        if not re.search(rf"(?<![\w-]){token}(?![\w-])", source_text, flags=re.IGNORECASE):
            ungrounded.append(slot)
    return ungrounded


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 3) if total else None


def main(
    trials_per_case: int = 5,
    *,
    call=None,
    limiter: RateLimiter | None = None,
    sleep=time.sleep,
    results_dir: Path | None = None,
) -> dict:
    templates = load_templates(REPO_ROOT / "agent" / "templates" / "questions.yaml")
    routing = load_routing(settings.model_routing_path)
    spec = spec_for("specialists", routing)
    limiter = limiter or RateLimiter(settings.requests_per_minute)
    counter = RequestCounter()
    call_kwargs = {} if call is None else {"call": call}

    records = []
    aborted_reason = None
    for incident, evidence, expected in CASES:
        if aborted_reason:
            break
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
                "ungrounded_identifiers": [],
                "specialist": None,
                "rendered": None,
                "usable": False,
            }
            try:
                decision = generate_structured(
                    prompt=prompt,
                    schema=SlotFillDecision,
                    role="specialists",
                    spec=spec,
                    limiter=limiter,
                    counter=counter,
                    sleep=sleep,
                    **call_kwargs,
                )
            except RetryableError as exc:
                aborted_reason = str(exc)
                break
            except ValueError as exc:
                record["parse_failed"] = True
                record["problems"] = [str(exc)[:300]]
            else:
                record["specialist"] = decision.specialist
                record["problems"] = validate_decision(decision, templates)
                if not record["problems"]:
                    record["ungrounded_identifiers"] = ungrounded_identifiers(
                        slot_mapping(decision, templates), f"{incident}\n{evidence}"
                    )
                    record["rendered"] = render(decision, templates)
                    record["usable"] = not record["ungrounded_identifiers"]
            records.append(record)

    total = len(records)
    summary = {
        "model": spec.model,
        "complete": aborted_reason is None,
        "aborted_reason": aborted_reason,
        "trials": total,
        "requests_made": counter.total,
        "parse_failure_rate": _rate(sum(r["parse_failed"] for r in records), total),
        "invalid_decision_rate": _rate(sum(bool(r["problems"]) for r in records), total),
        "ungrounded_identifier_rate": _rate(
            sum(bool(r["ungrounded_identifiers"]) for r in records), total
        ),
        "usable_rate": _rate(sum(r["usable"] for r in records), total),
        "specialist_match_rate": _rate(
            sum(r["specialist"] == r["expected_specialist"] for r in records), total
        ),
    }

    out_dir = results_dir or settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "spike-slot-fill.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if main()["complete"] else 1)
```

`requests_made` can exceed `trials` because retries are counted. Comparing the two after a live run shows how much of the budget went to rate limiting rather than to measurement.

Note that `specialist_match_rate` is softer evidence than the other numbers. The "expected" specialist is one reasonable choice rather than the only defensible one, so read it as a signal about whether routing is sensible, not as an accuracy score.

- [ ] **Step 10: Run the spike tests and the full suite**

Run: `.venv/bin/pytest tests/test_spike.py -v`
Expected: 3 passed.

Then: `.venv/bin/pytest -v`. Everything passes; integration tests are deselected.

- [ ] **Step 11: Commit the spike**

```bash
git add spikes/__init__.py spikes/slot_fill.py tests/test_spike.py
git commit -m "feat: slot-fill capability spike, grounded usability, offline-tested"
```

- [ ] **Step 12: Run the spike live, when a key is available**

This step needs `WATCHFLOOR_GOOGLE_API_KEY` in `.env`. Without it, stop here and report Steps 12 to 14 as deferred. Phase 0 cannot close until they are done.

Run: `.venv/bin/python -m spikes.slot_fill`
Expected: a JSON summary with `"complete": true`, exit code 0, and `results/spike-slot-fill.json` written. Twenty trials at twelve requests per minute take under two minutes.

If `complete` is false, the provider gave up partway. The partial file is kept for diagnosis, but **do not make the Phase 0 decision from an incomplete run.** Wait for the quota to reset and run it again.

- [ ] **Step 13: Record the result and decide**

Add to `DECISIONS.md` under a new dated heading, filling in the real numbers:

```markdown
## <date>, Phase 0 spike result

- Slot-fill spike on <model>: usable_rate <X>, ungrounded_identifier_rate <U>, parse_failure_rate <Y>, specialist_match_rate <Z> over 20 trials, <R> requests. Raw data in `results/spike-slot-fill.json`.
- Decision: <one of the three below>.
```

Read `usable_rate` against these thresholds. It already excludes ungrounded answers:

| `usable_rate` | What it means | What to do |
|---|---|---|
| 0.90 or above | The design in section 5.4 holds | Proceed to Phase 1 unchanged |
| 0.70 to 0.90 | Workable with a retry | Add one constrained retry that feeds the validation problems back in, record it as a decision, proceed |
| below 0.70 | The specialist tier is too weak for slot filling | Move specialists up to the flash tier in `config/models.yaml`, rerun the spike, and if it still fails, revisit section 5.4 before building Phase 1 |

This is the phase's whole purpose. Do not proceed to Phase 1 without an entry in `DECISIONS.md` recording which branch was taken.

- [ ] **Step 14: Commit the result**

```bash
git add DECISIONS.md results/spike-slot-fill.json
git commit -m "docs: Phase 0 spike result and decision"
```

---

## Phase 0 acceptance

All five must hold before Phase 1 begins.

1. `pytest` passes, and `pytest -m integration` passes against a live cluster.
2. The kind cluster comes up and tears down repeatably, creating it twice is a no-op, and `~/.kube/config` is byte-for-byte unchanged by any of it.
3. Scenario 001 breaks, and its `verify.sh` confirms both halves of its label: a checkout pod in `CreateContainerConfigError`, and no checkout pod Ready. Proven by the integration test in Task 4, with `label.yaml` unchanged since its pre-registration commit.
4. `results/spike-slot-fill.json` exists from a run whose summary says `"complete": true`.
5. `DECISIONS.md` records the spike's numbers and which branch of the threshold table was taken.

---

## What Phase 1 picks up

The MCP server with at least eight read-only tools, flat schemas, and recoverable error messages. Four more scenarios spanning the difficulty tiers. A manual tool-runner script. The acceptance gate is that a human can diagnose every scenario from the tool output alone, because if a person cannot, no agent will.
