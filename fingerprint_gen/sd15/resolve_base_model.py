"""Resolve local model directories or explicit Hugging Face model identifiers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from utils.run_logger import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PROJECT_ROOT / "configs/livdet2015/model_profile.json"

def _resolve(key: str, env: str, local_name: str, marker: str, profile_path=None) -> str:
    profile = json.loads(Path(profile_path or DEFAULT_PROFILE).read_text(encoding="utf-8"))
    explicit = os.environ.get(env, "").strip()
    candidates = [explicit] if explicit else [str(PROJECT_ROOT / "hf_models" / local_name)]
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if (path / marker).is_file():
            return str(path.resolve())
    if explicit:
        raise FileNotFoundError(f"{env} does not contain {marker}: {explicit}")
    return profile[key]

def resolve_base_model(profile_path=None) -> str:
    return _resolve("base_model", "LOCAL_MODEL_DIR", "stable-diffusion-v1-5", "model_index.json", profile_path)

def resolve_controlnet_model(profile_path=None) -> str:
    return _resolve("controlnet_model", "LOCAL_CONTROLNET_DIR", "sd-controlnet-canny", "config.json", profile_path)

def is_local_model_dir(path: str) -> bool:
    return (Path(path) / "model_index.json").is_file()

def model_identity(model: str) -> dict:
    root = Path(model)
    if root.is_dir():
        hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
                  if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)}
        return {"path": str(root.resolve()), "sha256": hashes}
    from huggingface_hub import model_info
    return {"repo_id": model, "revision": model_info(model).sha}

