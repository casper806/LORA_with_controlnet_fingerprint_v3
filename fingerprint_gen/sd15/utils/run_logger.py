"""Atomic run records and reproducibility fingerprints."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def software_versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
                 "datasets", "numpy", "Pillow", "opencv-python", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def weight_fingerprint(path: str | Path) -> dict:
    root = Path(path).resolve()
    if root.is_file():
        return {root.name: sha256_file(root)}
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"})
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise FileNotFoundError(f"No model weights in {root}")
    return {p.name: sha256_file(p) for p in files}

def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def write_train_config(output_dir: str, config: dict) -> str:
    path = Path(output_dir) / "train_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        previous = {k: v for k, v in existing.items() if k != "created_at"}
        if previous != config:
            raise ValueError(f"Training settings or inputs changed: {path}. Use a new output directory.")
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError(f"Untracked training output in {path.parent}. Use a new output directory.")
        save_json(path, {"created_at": utc_now_iso(), **config})
    return str(path)

def write_done_marker(output_dir: str, extra: dict | None = None) -> str:
    path = Path(output_dir) / "DONE"
    save_json(path, {"finished_at": utc_now_iso(), **(extra or {})})
    return str(path)

def is_done(output_dir: str) -> bool:
    return (Path(output_dir) / "DONE").is_file()

def training_inputs(metadata: str | Path, train_root: str | Path, combo: str | None = None) -> dict:
    root = Path(train_root).resolve()
    if root.name != "Training":
        raise ValueError("Training inputs must be rooted at the Training partition.")
    digest = hashlib.sha256()
    count = 0
    seen = set()
    with open(metadata, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in sorted(rows, key=lambda r: r["file_name"]):
        if combo and row.get("combo_key") != combo:
            continue
        path = (root / row["file_name"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid Training image: {row['file_name']}")
        if row["file_name"] in seen:
            raise ValueError(f"Duplicate Training image: {row['file_name']}")
        seen.add(row["file_name"])
        digest.update(json.dumps([row["file_name"], sha256_file(path)], separators=(",", ":")).encode())
        count += 1
    if not count:
        raise ValueError("No Training images were selected.")
    return {"count": count, "image_manifest_sha256": digest.hexdigest(), "metadata_sha256": sha256_file(metadata)}

