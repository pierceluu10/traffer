import json
import os
import shutil
from datetime import datetime
from typing import Dict, Optional

VERSIONS_DIR = "model/versions"
REGISTRY_FILE = "model/versions/registry.json"


def _load_registry() -> Dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"models": [], "current": None}


def _save_registry(registry: Dict):
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(version: str, weights_path: str, metrics: Dict, notes: str = "") -> Dict:
    registry = _load_registry()
    version_dir = os.path.join(VERSIONS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy2(weights_path, os.path.join(version_dir, "weights.pth"))

    entry = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "notes": notes,
        "path": version_dir
    }
    registry["models"].append(entry)
    registry["current"] = version
    _save_registry(registry)
    return entry


def rollback(version: str) -> Optional[Dict]:
    registry = _load_registry()
    for entry in registry["models"]:
        if entry["version"] == version:
            registry["current"] = version
            _save_registry(registry)
            return entry
    return None


def get_current_version() -> Optional[str]:
    registry = _load_registry()
    return registry.get("current")


def list_versions() -> list:
    registry = _load_registry()
    return registry.get("models", [])
