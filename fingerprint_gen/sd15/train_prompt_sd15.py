#!/usr/bin/env python3
"""Train one caption-anchored residual PromptLearner with a frozen shared LoRA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from build_livdet_metadata import make_combo_text
from resolve_base_model import resolve_base_model, model_identity
from utils.run_logger import (sha256_file, software_versions, weight_fingerprint, save_json,
                              write_train_config, write_done_marker, is_done, training_inputs)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "fingerprint_gen/train_residual_prompt.py"

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("combo_key", "train_data_dir", "metadata_jsonl", "class_index_json", "lora_path", "output_dir", "lexicon_path"):
        p.add_argument("--" + name, required=True)
    for name, value in {"resolution": 512, "seed": 42, "train_batch_size": 4, "gradient_accumulation_steps": 2,
                        "max_train_steps": 20000, "checkpointing_steps": 4000, "residual_rank": 8}.items():
        p.add_argument("--" + name, type=int, default=value)
    for name, value in {"learning_rate": 1e-3, "residual_scale": 1.0, "residual_kappa": 0.25,
                        "anchor_loss_weight": 1.0, "residual_loss_weight": 1e-4, "cosine_min": 0.97,
                        "norm_ratio_min": 0.9, "norm_ratio_max": 1.1, "lora_scale": 0.8}.items():
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    lexicon = json.loads(Path(args.lexicon_path).read_text(encoding="utf-8"))
    with open(args.metadata_jsonl, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [{**r, "class_id": 0} for r in rows if r.get("combo_key") == args.combo_key]
    if not rows:
        raise ValueError(f"No Training samples for {args.combo_key}")
    caption = make_combo_text(lexicon, rows[0]["sensor"], rows[0]["material"])
    if any(r.get("init_prompt") != caption for r in rows):
        raise ValueError("Metadata captions differ from the selected lexicon. Rebuild metadata first.")
    base_model = resolve_base_model()
    base_identity = model_identity(base_model)
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    config.update(base_model=base_identity, caption=caption, software=software_versions(),
                  lexicon_sha256=sha256_file(args.lexicon_path), index_sha256=sha256_file(args.class_index_json),
                  training_inputs=training_inputs(args.metadata_jsonl, args.train_data_dir, args.combo_key),
                  lora_weights=weight_fingerprint(args.lora_path), trainer_sha256=sha256_file(TRAIN_SCRIPT),
                  learner_sha256=sha256_file(TRAIN_SCRIPT.with_name("prompt_learner.py")))
    write_train_config(str(output), config)
    final = output / "final_prompt_learner.pt"
    if is_done(str(output)):
        done = json.loads((output / "DONE").read_text(encoding="utf-8"))
        if not final.is_file() or sha256_file(final) != done["artifact_sha256"]:
            raise ValueError("Completed PromptLearner weights are missing or changed.")
        print(f"Verified completed PromptLearner: {output}")
        return
    checkpoints = sorted((p for p in output.glob("checkpoint-*") if p.is_dir() and (p / "progress.json").is_file()), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    interrupted = final.exists() or (output / "train.log").exists()
    if args.resume and not checkpoints and interrupted:
        raise ValueError("No full PromptLearner checkpoint is available to resume.")
    if not args.resume and (checkpoints or final.exists() or (output / "train.log").exists()):
        raise ValueError("Training output already exists. Use --resume or a new output directory.")
    filtered = output / "metadata_filtered.jsonl"
    filtered.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    index = output / "class_index_filtered.json"
    save_json(index, {"num_classes": 1, "style_key_to_id": {args.combo_key: 0},
                      "id_to_style_key": {"0": args.combo_key}, "id_to_init_prompt": {"0": caption}})
    excluded = {"combo_key", "lexicon_path", "resume"}
    values = {k: v for k, v in vars(args).items() if k not in excluded}
    values.update(pretrained_model_name_or_path=base_model, metadata_jsonl=filtered, class_index_json=index,
                  report_to="tensorboard")
    if "revision" in base_identity:
        values["revision"] = base_identity["revision"]
    cmd = [sys.executable, str(TRAIN_SCRIPT), *[f"--{k}={v}" for k, v in values.items()], "--allow_tf32"]
    if args.resume and checkpoints:
        cmd.append("--resume_from_checkpoint=latest")
    print(f"Training {args.combo_key}; log: {output / 'train.log'}", flush=True)
    with (output / "train.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": cmd}) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not final.is_file():
        raise RuntimeError(f"Training exited without final PromptLearner weights: {final}")
    write_done_marker(str(output), {"artifact_sha256": sha256_file(final), "step": args.max_train_steps})
    print(f"Saved PromptLearner: {final}")

if __name__ == "__main__":
    main()
