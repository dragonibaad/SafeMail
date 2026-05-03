from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Centralized runtime paths and constants."""

    project_root: Path = Path(__file__).resolve().parents[2]
    dataset_path: Path = project_root / "Email Dataset" / "Email_Dataset.csv"
    models_dir: Path = project_root / "Models"
    random_state: int = 42
    test_size: float = 0.2

    @classmethod
    def from_env(cls) -> "AppConfig":
        instance = cls()
        dataset_override = os.getenv("SAFEMAIL_DATASET_PATH")
        models_override = os.getenv("SAFEMAIL_MODELS_DIR")
        random_state = os.getenv("SAFEMAIL_RANDOM_STATE")
        test_size = os.getenv("SAFEMAIL_TEST_SIZE")
        return cls(
            project_root=instance.project_root,
            dataset_path=Path(dataset_override) if dataset_override else instance.dataset_path,
            models_dir=Path(models_override) if models_override else instance.models_dir,
            random_state=int(random_state) if random_state else instance.random_state,
            test_size=float(test_size) if test_size else instance.test_size,
        )
