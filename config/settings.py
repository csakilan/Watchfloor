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
