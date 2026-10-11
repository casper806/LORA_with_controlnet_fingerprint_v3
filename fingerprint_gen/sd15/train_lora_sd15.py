#!/usr/bin/env python3
"""Train the shared SD1.5 LoRA on the Training partition."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from resolve_base_model import resolve_base_model, model_identity
from utils.run_logger import (sha256_file, software_versions,
                              write_train_config, write_done_marker, is_done, training_inputs)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_SCRIPT = PROJECT_ROOT / "diffusers/examples/text_to_image/train_text_to_image_lora.py"

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", default="shared")
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--metadata_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max_train_steps", type=int, default=15000)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--checkpointing_steps", type=int, default=2000)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    train_root = Path(args.train_data_dir).resolve()
    output = Path(args.output_dir).resolve()
    base_model = resolve_base_model()
    base_identity = model_identity(base_model)
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    config.update(base_model=base_identity, software=software_versions(),
                  training_inputs=training_inputs(args.metadata_jsonl, train_root),
                  trainer_sha256=sha256_file(LORA_SCRIPT))
    write_train_config(str(output), config)
    final = output / "pytorch_lora_weights.safetensors"
    if is_done(str(output)):
        done = json.loads((output / "DONE").read_text(encoding="utf-8"))
        if not final.is_file() or sha256_file(final) != done["artifact_sha256"]:
            raise ValueError("Completed LoRA weights are missing or changed.")
        print(f"Verified completed LoRA: {output}")
        return
    checkpoints = sorted((p for p in output.glob("checkpoint-*") if p.is_dir() and (p / "progress.json").is_file()), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    interrupted = final.exists() or (output / "train.log").exists()
    if args.resume and not checkpoints and interrupted:
        raise ValueError("No full LoRA checkpoint is available to resume.")
    if not args.resume and (checkpoints or final.exists() or (output / "train.log").exists()):
        raise ValueError("Training output already exists. Use --resume or a new output directory.")
    with tempfile.TemporaryDirectory(prefix="livdet2015_lora_") as temp:
        prepared = Path(temp)
        shutil.copy2(args.metadata_jsonl, prepared / "metadata.jsonl")
        with open(args.metadata_jsonl, encoding="utf-8") as stream:
            sensors = sorted({Path(json.loads(line)["file_name"]).parts[0] for line in stream if line.strip()})
        for sensor in sensors:
            os.symlink(train_root / sensor, prepared / sensor, target_is_directory=True)
        values = {k: v for k, v in vars(args).items() if k not in {"run_id", "metadata_jsonl", "resume"}}
        values.update(pretrained_model_name_or_path=base_model, train_data_dir=prepared,
                      caption_column="text", lr_scheduler="constant", lr_warmup_steps=0,
                      report_to="tensorboard", logging_dir=str(output / "logs"))
        if "revision" in base_identity:
            values["revision"] = base_identity["revision"]
        cmd = [sys.executable, str(LORA_SCRIPT), *[f"--{k}={v}" for k, v in values.items()],
               "--center_crop", "--gradient_checkpointing"]
        if args.resume and checkpoints:
            cmd.append(f"--resume_from_checkpoint={checkpoints[-1]}")
        print(f"Training shared LoRA; log: {output / 'train.log'}", flush=True)
        with (output / "train.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": cmd}) + "\n")
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not final.is_file():
        raise RuntimeError(f"Training exited without final LoRA weights: {final}")
    write_done_marker(str(output), {"artifact_sha256": sha256_file(final), "step": args.max_train_steps})
    print(f"Saved LoRA: {final}")

if __name__ == "__main__":
    main()
