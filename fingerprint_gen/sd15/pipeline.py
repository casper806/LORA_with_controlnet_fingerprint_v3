#!/usr/bin/env python3
"""Run the configured LivDet2015 metadata, training, and generation stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SD15 = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["metadata", "lora", "prompt", "generate", "all"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/livdet2015/defaults.json")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("LIVDET2015_ROOT", PROJECT_ROOT / "data/livdet2015"))
    parser.add_argument("--weight-root", type=Path, default=os.environ.get("WEIGHT_ROOT", PROJECT_ROOT / "weights/livdet2015"))
    parser.add_argument("--result-root", type=Path, default=os.environ.get("RESULT_ROOT", PROJECT_ROOT / "results/livdet2015"))
    parser.add_argument("--lexicon", type=Path, default=os.environ.get("PROMPT_LEXICON", PROJECT_ROOT / "configs/livdet2015/prompt_lexicon.json"))
    parser.add_argument("--seed", type=int, default=int(os.environ["SEED"]) if "SEED" in os.environ else None)
    parser.add_argument("--generation-seed", type=int, help="Override only the inference seed; keep the trained checkpoints fixed.")
    parser.add_argument("--run-id", default=os.environ.get("INFERENCE_RUN_ID", "generation"))
    parser.add_argument("--lora-run-id", default=os.environ.get("LORA_RUN_ID", "shared"))
    parser.add_argument("--split", choices=["Training", "Testing", "both"], default="both")
    parser.add_argument("--sensor", action="append", help="Select a sensor; may be repeated.")
    parser.add_argument("--combo", action="append", help="Select a sensor-material combination; may be repeated.")
    parser.add_argument("--max-images", type=int, help="Limit sources per sensor and split for a small run.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a full state checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training, generating, or creating artifacts.")
    return parser.parse_args()


def options(values: dict) -> list[str]:
    return [f"--{key}={value}" for key, value in values.items()]


def selected_combos(config, sensors=None, combos=None):
    mapping = config["combos"]
    if sensors and set(sensors) - set(mapping):
        raise ValueError(f"Unknown sensors: {sorted(set(sensors) - set(mapping))}")
    available = {ck for sensor, keys in mapping.items() if not sensors or sensor in sensors for ck in keys}
    if combos and set(combos) - available:
        raise ValueError(f"Unknown or excluded combinations: {sorted(set(combos) - available)}")
    return {s: [ck for ck in keys if not combos or ck in combos]
            for s, keys in mapping.items() if (not sensors or s in sensors) and any(not combos or ck in combos for ck in keys)}


def build_commands(args, config):
    data = args.data_root.resolve()
    weights = args.weight_root.resolve()
    results = args.result_root.resolve()
    seed = config["seed"] if args.seed is None else args.seed
    generation_seed = seed if args.generation_seed is None else args.generation_seed
    combos = selected_combos(config, args.sensor, args.combo)
    for name in (args.run_id, args.lora_run_id):
        if not name or Path(name).name != name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Run IDs must be single directory names.")
    stages = ["metadata", "lora", "prompt", "generate"] if args.stage == "all" else [args.stage]
    lora_dir = weights / "lora" / args.lora_run_id
    meta = weights / "metadata"
    commands = []
    common = {"resolution": config["resolution"], "seed": seed}
    if "metadata" in stages:
        commands.append([sys.executable, str(SD15 / "build_livdet_metadata.py"),
                         *options({"root_dir": data, "out_dir": meta, "lexicon_path": args.lexicon.resolve()})])
    if "lora" in stages:
        commands.append([sys.executable, str(SD15 / "train_lora_sd15.py"),
                         *options({**common, **config["lora"], "run_id": args.lora_run_id,
                                   "train_data_dir": data / "Training", "metadata_jsonl": meta / "metadata_lora_all.jsonl",
                                   "output_dir": lora_dir}), *(["--resume"] if args.resume else [])])
    if "prompt" in stages:
        for keys in combos.values():
            for ck in keys:
                commands.append([sys.executable, str(SD15 / "train_prompt_sd15.py"),
                                 *options({**common, **config["prompt"], "combo_key": ck,
                                           "train_data_dir": data / "Training", "metadata_jsonl": meta / "metadata_prompt_combo.jsonl",
                                           "class_index_json": meta / "class_index_prompt_combo.json",
                                           "lexicon_path": args.lexicon.resolve(), "lora_path": lora_dir,
                                           "output_dir": weights / "prompt_combo" / ck}), *(["--resume"] if args.resume else [])])
    if "generate" in stages:
        splits = ["Training", "Testing"] if args.split == "both" else [args.split]
        for split in splits:
            for sensor, keys in combos.items():
                values = {"resolution": config["resolution"], "seed": generation_seed, **config["inference"],
                          "layout_root": data, "split": split, "sensor": sensor,
                          "target_combo_keys": ",".join(keys), "lora_path": lora_dir,
                          "prompt_combo_root": weights / "prompt_combo",
                          "output_dir": results / args.run_id / split / sensor}
                if args.max_images is not None:
                    values["max_images"] = args.max_images
                commands.append([sys.executable, str(SD15 / "batch_gen_livdet_sd15.py"), *options(values)])
    return commands


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commands = build_commands(args, config)
    if not args.dry_run:
        required_splits = set()
        if args.stage in {"metadata", "lora", "prompt", "all"}:
            required_splits.add("Training")
        if args.stage in {"generate", "all"}:
            required_splits.update(["Training", "Testing"] if args.split == "both" else [args.split])
        for split in required_splits:
            if not (args.data_root / split).is_dir():
                raise FileNotFoundError(f"Missing configured dataset split: {args.data_root / split}")
        if not args.lexicon.is_file():
            raise FileNotFoundError(f"Missing lexicon: {args.lexicon}")
    environment = {**os.environ, "WEIGHT_ROOT": str(args.weight_root.resolve()),
                   "RESULT_ROOT": str(args.result_root.resolve()), "LIVDET2015_ROOT": str(args.data_root.resolve()),
                   "PROMPT_LEXICON": str(args.lexicon.resolve()), "LORA_RUN_ID": args.lora_run_id}
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, env=environment, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
