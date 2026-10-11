#!/usr/bin/env python3
"""Generate sensor-material spoof images from bona fide sources in one split."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "fingerprint_gen"))
from resolve_base_model import resolve_base_model, resolve_controlnet_model, model_identity
from run_manifest import RunManifest
from utils.run_logger import save_json, sha256_file, software_versions, weight_fingerprint

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout_root", type=Path, required=True)
    p.add_argument("--split", choices=["Training", "Testing"], required=True)
    p.add_argument("--sensor", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--lora_path", type=Path, required=True)
    p.add_argument("--prompt_combo_root", type=Path, required=True)
    p.add_argument("--target_combo_keys", required=True, help="Comma-separated sensor-material keys.")
    for name, value in {"resolution": 512, "inference_steps": 30, "canny_low": 50, "canny_high": 200, "seed": 42}.items():
        p.add_argument("--" + name, type=int, default=value)
    for name, value in {"guidance_scale": 7.0, "control_scale": 0.6, "strength": 0.92, "lora_scale": 0.8}.items():
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--max_images", type=int)
    return p.parse_args()

def derived_seed(seed: int, combo: str, stem: str) -> int:
    return seed + zlib.adler32(combo.encode("utf-8")) + zlib.adler32(stem.encode("utf-8"))

def collect_sources(root: Path, split: str, sensor: str, limit: int | None = None) -> list[Path]:
    sensor_root = (root / split / sensor).resolve()
    if not sensor_root.is_relative_to(root.resolve()) or not sensor_root.is_dir():
        raise FileNotFoundError(f"Missing sensor directory: {sensor_root}")
    sources = []
    for path in sorted(sensor_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = [p.lower() for p in path.relative_to(sensor_root).parts[:-1]]
        if any(p.startswith("@") or p in {"generated", "__pycache__"} for p in parts):
            continue
        if path.name.startswith(("._", "SYNOFILE")):
            continue
        if "live" not in parts or "fake" in parts:
            continue
        if not path.resolve().is_relative_to(sensor_root):
            raise ValueError(f"Source symlink leaves its sensor split: {path}")
        sources.append(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--max_images must be positive.")
        sources = sources[:limit]
    if not sources:
        raise ValueError(f"No bona fide images under {sensor_root}; expected live/ or {sensor}/Live/.")
    return sources

def build_plan(args):
    root = args.layout_root.resolve()
    sensor_root = root / args.split / args.sensor
    if not 0 <= args.seed < 2**32:
        raise ValueError("Base seed must be in [0, 2**32).")
    keys = args.target_combo_keys.replace(",", " ").split()
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Specify unique target combination keys.")
    for key in keys:
        if not key.startswith(args.sensor + "_") or Path(key).name != key or "/" in key or "\\" in key:
            raise ValueError(f"Invalid sensor-material combination: {key}")
        checkpoint = args.prompt_combo_root / key / "final_prompt_learner.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing PromptLearner: {checkpoint}")
    sources = collect_sources(root, args.split, args.sensor, args.max_images)
    jobs = []
    for source in sources:
        rel = source.relative_to(root).as_posix()
        source_hash = sha256_file(source)
        for key in keys:
            output = (Path(key) / source.relative_to(sensor_root).parent / (source.name + "_gen.png")).as_posix()
            jobs.append({"source": rel, "source_sha256": source_hash, "split": args.split,
                         "sensor": args.sensor, "combo_key": key, "output": output,
                         "seed": derived_seed(args.seed, key, source.stem)})
    outputs = [job["output"] for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Multiple source images map to the same output.")
    return keys, jobs

def main():
    args = parse_args()
    if args.resolution < 8 or args.resolution % 8 or args.inference_steps < 1:
        raise ValueError("Resolution must be a positive multiple of 8 and inference_steps must be positive.")
    if not 0 < args.strength <= 1 or not 0 <= args.canny_low < args.canny_high <= 255:
        raise ValueError("Invalid Img2Img strength or Canny thresholds.")
    keys, jobs = build_plan(args)
    base_model = resolve_base_model()
    controlnet_model = resolve_controlnet_model()
    base_identity = model_identity(base_model)
    control_identity = model_identity(controlnet_model)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"output_dir", "target_combo_keys"}}
    config = {"settings": settings, "target_combo_keys": keys, "sources": jobs,
              "base_model": base_identity, "controlnet": control_identity,
              "lora_weights": weight_fingerprint(args.lora_path),
              "prompt_weights": {key: sha256_file(args.prompt_combo_root / key / "final_prompt_learner.pt") for key in keys},
              "software": software_versions(), "generator_sha256": sha256_file(__file__),
              "learner_sha256": sha256_file(PROJECT_ROOT / "fingerprint_gen/prompt_learner.py"),
              "force_grayscale": True, "seed_rule": "base + Adler32(combo) + Adler32(source_stem)"}
    with RunManifest(args.output_dir, config) as manifest:
        pending = [job for job in jobs if not manifest.completed(job)]
        save_json(manifest.root / "generation_plan.json",
                  {"split": args.split, "source_count": len(jobs) // len(keys), "output_count": len(jobs),
                   "target_combo_keys": keys})
        if not pending:
            print(f"All {len(jobs)} outputs and manifest records verified.")
            return
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from tqdm import tqdm
        from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
        from prompt_learner import PromptLearner

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=dtype, local_files_only=Path(controlnet_model).is_dir(),
            **({"revision": control_identity["revision"]} if "revision" in control_identity else {}))
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model, controlnet=controlnet, safety_checker=None, torch_dtype=dtype,
            local_files_only=Path(base_model).is_dir(),
            **({"revision": base_identity["revision"]} if "revision" in base_identity else {})).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
        pipe.load_lora_weights(str(args.lora_path), adapter_name="fingerprint_lora")
        pipe.set_adapters(["fingerprint_lora"], adapter_weights=[args.lora_scale])
        learners = {key: PromptLearner.from_checkpoint(
            str(args.prompt_combo_root / key / "final_prompt_learner.pt"), device=device, dtype=dtype).eval()
            for key in keys}
        for job in tqdm(pending, desc=f"{args.split}/{args.sensor}"):
            with Image.open(args.layout_root / job["source"]) as original:
                init_image = original.convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
            canny = Image.fromarray(cv2.Canny(np.asarray(init_image), args.canny_low, args.canny_high)).convert("RGB")
            generator = torch.Generator(device=device).manual_seed(job["seed"])
            with torch.inference_mode():
                result = pipe(image=init_image, control_image=canny,
                              prompt_embeds=learners[job["combo_key"]].condition(0, device=device, dtype=dtype),
                              controlnet_conditioning_scale=args.control_scale, strength=args.strength,
                              num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale,
                              height=args.resolution, width=args.resolution, generator=generator)
            output = manifest.output_path(job["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            result.images[0].convert("L").save(temporary, format="PNG")
            os.replace(temporary, output)
            manifest.append(job)
        print(f"Verified {len(manifest.records)} outputs: {manifest.root}")

if __name__ == "__main__":
    main()

