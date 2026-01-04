from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:

    APP_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = APP_DIR / "keystroke_stress_model.joblib"
    PAUSE_THRESHOLD_S: float = 1.0  
    STRESS_PROB_THRESHOLD: float = 0.5


settings = Settings()
