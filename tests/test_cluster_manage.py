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
