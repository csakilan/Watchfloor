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
