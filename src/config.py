import os
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


def get_mode() -> str:
    mode = os.getenv("COURSEMIND_MODE", "mock").strip().lower()
    return mode if mode in {"mock", "hybrid", "real"} else "mock"


def get_bandit_state_path() -> Path:
    return Path(os.getenv("BANDIT_STATE_PATH", "data/processed/quiz_state.json"))
