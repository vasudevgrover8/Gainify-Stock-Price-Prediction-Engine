import json
from pathlib import Path


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_read_json(path, default=None):
    json_path = Path(path)
    if not json_path.exists():
        return default
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def safe_write_json(path, data):
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return json_path
