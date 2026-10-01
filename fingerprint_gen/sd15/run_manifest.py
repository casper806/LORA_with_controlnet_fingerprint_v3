"""Preserve generation provenance across interrupted and repeated runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from utils.run_logger import save_json, sha256_file


class RunManifest:
    def __init__(self, output_dir: str | Path, config: dict):
        self.root = Path(output_dir).resolve()
        self.config = config
        self.records = {}
        self.stream = None
        self.lock = self.root / ".run.lock"

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Run is locked: {self.lock}. If its process has stopped, remove only this lock file.") from exc
        with os.fdopen(descriptor, "w") as lock:
            lock.write(str(os.getpid()))
        try:
            config_path = self.root / "inference_config.json"
            manifest = self.root / "generation_manifest.jsonl"
            if config_path.exists():
                if json.loads(config_path.read_text(encoding="utf-8")) != self.config:
                    raise ValueError("Generation settings, sources, weights, or software changed. Use a new output directory.")
            else:
                if any(p != self.lock for p in self.root.iterdir()):
                    raise ValueError("Output directory contains untracked files. Use a new output directory.")
                save_json(config_path, self.config)
            if manifest.exists():
                with manifest.open(encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record["output"] in self.records:
                            raise ValueError(f"Duplicate manifest output: {record['output']}")
                        self.records[record["output"]] = record
            self.stream = manifest.open("a", encoding="utf-8")
            return self
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def output_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Output escapes run directory: {relative}")
        return path

    def completed(self, record: dict) -> bool:
        relative = record["output"]
        path = self.output_path(relative)
        previous = self.records.get(relative)
        if previous:
            if any(previous.get(k) != value for k, value in record.items()):
                raise ValueError(f"Manifest identity changed: {relative}")
            if not path.is_file() or sha256_file(path) != previous["output_sha256"]:
                raise ValueError(f"Recorded output missing or modified: {relative}. Use a new output directory.")
            return True
        if path.exists():
            raise ValueError(f"Unrecorded output: {relative}. Use a new output directory or remove this orphan file.")
        return False

    def append(self, record: dict) -> None:
        if record["output"] in self.records:
            raise ValueError(f"Output already recorded: {record['output']}")
        record = {**record, "output_sha256": sha256_file(self.output_path(record["output"]))}
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records[record["output"]] = record

    def __exit__(self, *args):
        if self.stream:
            self.stream.close()
        self.lock.unlink(missing_ok=True)
