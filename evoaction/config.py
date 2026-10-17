"""Runtime settings, read from the environment (and `.env`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    api_base: str = field(default_factory=lambda: os.environ.get("CRUSOE_API_BASE", "https://api.inference.crusoecloud.com/v1"))
    reasoning_model: str = field(default_factory=lambda: os.environ.get("CRUSOE_REASONING_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"))
    vision_model: str = field(default_factory=lambda: os.environ.get("CRUSOE_VISION_MODEL", "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B"))

    max_nemotron_calls: int = field(default_factory=lambda: _int("MAX_NEMOTRON_CALLS", 24))
    max_vision_calls: int = field(default_factory=lambda: _int("MAX_VISION_CALLS", 48))
    max_agent_tasks: int = field(default_factory=lambda: _int("MAX_AGENT_TASKS", 12))
    max_candidate_experiments: int = field(default_factory=lambda: _int("MAX_CANDIDATE_EXPERIMENTS", 2))
    max_retries_per_call: int = field(default_factory=lambda: _int("MAX_RETRIES_PER_CALL", 1))
    max_spend_usd: float = field(default_factory=lambda: _float("MAX_SPEND_USD", 35.0))
    reserved_test_vision_calls: int = field(default_factory=lambda: _int("RESERVED_TEST_VISION_CALLS", 9))
    request_timeout_seconds: int = field(default_factory=lambda: _int("REQUEST_TIMEOUT_SECONDS", 180))

    artifacts_dir: Path = ROOT / "artifacts"
    cache_dir: Path = ROOT / "data" / "cache"

    @property
    def api_key(self) -> str | None:
        return os.environ.get("CRUSOE_API_KEY")

    @property
    def db_path(self) -> Path:
        return self.artifacts_dir / "evoaction.db"

    @property
    def events_path(self) -> Path:
        return self.artifacts_dir / "events.jsonl"

    @property
    def episodes_dir(self) -> Path:
        # Agent-visible media: video, telemetry, instruction. Never gold.
        return self.cache_dir / "episodes"

    @property
    def protected_dir(self) -> Path:
        # Evaluator-only reference labels. No agent tool receives this path.
        return self.cache_dir / "protected"


def get_settings() -> Settings:
    return Settings()
