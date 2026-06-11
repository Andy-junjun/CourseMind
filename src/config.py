import os
from pathlib import Path


def get_mode() -> str:
    mode = os.getenv("COURSEMIND_MODE", "mock").strip().lower()
    return mode if mode in {"mock", "hybrid", "real"} else "mock"


def get_bandit_state_path() -> Path:
    return Path(os.getenv("BANDIT_STATE_PATH", "data/processed/quiz_state.json"))

