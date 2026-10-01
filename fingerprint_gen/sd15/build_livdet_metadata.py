#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Training-only LivDet2015 metadata for the SD1.5 generator.

Output files:
  - metadata_lora_all.jsonl
  - metadata_prompt_combo.jsonl
  - class_index_lora_all.json
  - class_index_prompt_combo.json
  - combo_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "livdet2015"
DEFAULT_OUT = PROJECT_ROOT / "weights" / "livdet2015" / "metadata"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_paths import DEFAULT_LEXICON  # noqa: E402
from utils.run_logger import sha256_file  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
SKIP_DIR_NAMES = {"@eadir", "@eaDir", ".ds_store", "generated", "__pycache__"}
EXCLUDE_TOP_LEVEL_SENSORS = {"Time_Series"}


def sanitize_combo_part(name: str) -> str:
    s = name.strip().replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    return s


def combo_key(sensor: str, material: str) -> str:
    return f"{sensor}_{sanitize_combo_part(material)}"


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMG_EXTS)


def try_open_image(path: str) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB")
        return True, None
    except Exception as e:
        return False, str(e)


def load_lexicon(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_training_path(parts: list[str]) -> tuple[str, str, str, str, str]:
    if not parts:
        raise ValueError("empty path")
    lower = [p.lower() for p in parts]
    sensor = parts[0]
    authenticity = "unknown"
    auth_idx = None
    for i, p in enumerate(lower):
        if p == "live":
            authenticity = "live"
            auth_idx = i
            break
        if p == "fake":
            authenticity = "fake"
            auth_idx = i
            break
    fake_sub = ""
    if authenticity == "fake" and auth_idx is not None and auth_idx + 1 < len(parts) - 1:
        fake_sub = parts[auth_idx + 1]
    style_key = f"{sensor}|{authenticity}"
    if fake_sub:
        style_key += f"|{fake_sub}"
    path_hint = "/".join(parts[:-1]) if parts else ""
    return sensor, authenticity, fake_sub, style_key, path_hint


def material_key(fake_sub: str) -> str:
    return fake_sub.split("/")[0] if fake_sub else ""


def lookup_desc(table: dict, key: str, fallback: str) -> str:
    if key in table:
        return table[key]
    for k, v in table.items():
        if k.lower() == key.lower():
            return v
    return fallback


def make_lora_text(lex: dict) -> str:
    base = lex.get("lora_base", "fingerprint ridge pattern")
    return f"{base}, LivDet2015"


def make_combo_prefix(lex: dict, sensor: str, mat: str) -> str:
    """Compose the material and sensor appearance description."""
    ck = combo_key(sensor, mat)
    overrides = lex.get("combo_overrides", {})
    if ck in overrides:
        return overrides[ck]

    sensor_desc = lookup_desc(
        lex.get("sensor", {}),
        sensor,
        f"{sensor} live-scan",
    )
    mat_desc = lookup_desc(
        lex.get("material", {}),
        mat,
        f"spoof {mat} ridge-valley artifacts",
    )
    # Place material first so it is retained if CLIP truncates a long caption.
    return ", ".join([mat_desc, sensor_desc])


def make_combo_short_suffix(lex: dict) -> str:
    return lex.get(
        "prompt_suffix_combo_short",
        lex.get("prompt_suffix_combo", "PAD spoof, grayscale only"),
    )


def combo_suffix_mode(lex: dict, combo_key: str | None = None) -> str:
    overrides = lex.get("combo_train_suffix", {})
    if combo_key and combo_key in overrides:
        return overrides[combo_key]
    return lex.get("train_suffix", "short")


def make_combo_suffix(lex: dict, combo_key: str | None = None) -> str:
    if combo_suffix_mode(lex, combo_key) == "long":
        return lex.get(
            "prompt_suffix_combo",
            "LivDet2015 PAD spoof scan, live-to-fake material style transfer, grayscale only, no finger no color",
        )
    return make_combo_short_suffix(lex)


def make_combo_text(lex: dict, sensor: str, mat: str) -> str:
    """Compose the same caption for metadata and CLIP conditioning."""
    ck = combo_key(sensor, mat)
    return ", ".join([make_combo_prefix(lex, sensor, mat), make_combo_suffix(lex, ck)])


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_class_index(
    path: str,
    *,
    root: str,
    variant: str,
    style_to_id: OrderedDict,
    num_images: int,
    corrupt_count: int,
    id_to_init_prompt: dict[int, str] | None = None,
) -> None:
    payload = {
        "dataset": "LivDet2015",
        "variant": variant,
        "includes_only": "Training",
        "file_name_relative_to": "Training",
        "train_data_dir_hint": os.path.join(root, "Training"),
        "excludes_sensors": ["Time_Series"],
        "root_dir": root,
        "style_key_to_id": dict(style_to_id),
        "id_to_style_key": {str(v): k for k, v in style_to_id.items()},
        "num_classes": len(style_to_id),
        "num_images": num_images,
        "num_corrupt": corrupt_count,
    }
    if id_to_init_prompt:
        payload["id_to_init_prompt"] = {str(k): v for k, v in id_to_init_prompt.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description="Build Training-only LivDet2015 metadata.")
    p.add_argument("--root_dir", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--lexicon_path", type=str, default=str(DEFAULT_LEXICON))
    args = p.parse_args()

    root = os.path.abspath(args.root_dir)
    train_root = os.path.join(root, "Training")
    if not os.path.isdir(train_root):
        print(f"[ERROR] Missing Training folder: {train_root}", file=sys.stderr)
        return 1

    lex = load_lexicon(Path(args.lexicon_path))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    corrupt: list[dict] = []
    rel_paths: list[str] = []

    for dirpath, dirnames, files in os.walk(train_root):
        if os.path.relpath(dirpath, train_root).replace("\\", "/") == ".":
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIR_NAMES
                and not d.startswith("@")
                and d not in EXCLUDE_TOP_LEVEL_SENSORS
            ]
        else:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith("@")]

        for f in files:
            if f.startswith(("._", "SYNOFILE")):
                continue
            full = os.path.join(dirpath, f)
            if not is_image_file(full):
                continue
            rel = os.path.relpath(full, train_root).replace("\\", "/")
            ok, err = try_open_image(full)
            if not ok:
                corrupt.append({"path": full, "rel": rel, "error": err})
                continue
            rel_paths.append(rel)

    if corrupt:
        raise ValueError(f"Found {len(corrupt)} unreadable Training images. First: {corrupt[0]['rel']}")
    if not rel_paths:
        raise ValueError("No Training images were found.")
    rel_paths.sort()
    lora_records: list[dict] = []
    combo_style_to_id: OrderedDict[str, int] = OrderedDict()
    combo_records: list[dict] = []
    combo_init: dict[int, str] = {}
    combo_manifest: list[dict] = []

    lora_text = make_lora_text(lex)

    for rel in rel_paths:
        parts = rel.split("/")
        try:
            sensor, authenticity, fake_sub, style_key, path_hint = parse_training_path(parts)
        except ValueError as e:
            print(f"[WARN] skip {rel}: {e}", file=sys.stderr)
            continue

        if authenticity not in {"live", "fake"}:
            raise ValueError(f"Expected live/ or fake/ in Training path: {rel}")
        image_path = Path(train_root) / rel
        if not image_path.resolve().is_relative_to(Path(train_root).resolve()):
            raise ValueError(f"Training image leaves the Training split: {rel}")
        image_hash = sha256_file(image_path)
        lora_records.append(
            {
                "file_name": rel,
                "sha256": image_hash,
                "text": lora_text,
                "init_prompt": lora_text,
                "class_id": 0,
                "dataset": "LivDet2015",
                "sensor": sensor,
                "authenticity": authenticity,
                "fake_subcategory": fake_sub,
                "path_hint": path_hint,
                "style_key": style_key,
            }
        )

        if authenticity != "fake":
            continue

        mat = material_key(fake_sub)
        if not mat:
            raise ValueError(f"Spoof image has no material subdirectory: {rel}")

        ck = combo_key(sensor, mat)
        if ck not in combo_style_to_id:
            cid = len(combo_style_to_id)
            combo_style_to_id[ck] = cid
            init = make_combo_text(lex, sensor, mat)
            combo_init[cid] = init
            combo_manifest.append(
                {
                    "combo_key": ck,
                    "class_id": cid,
                    "sensor": sensor,
                    "material": mat,
                    "init_prompt": init,
                }
            )

        combo_cid = combo_style_to_id[ck]
        init_prompt = combo_init[combo_cid]
        combo_records.append(
            {
                "file_name": rel,
                "sha256": image_hash,
                "text": init_prompt,
                "init_prompt": init_prompt,
                "class_id": combo_cid,
                "dataset": "LivDet2015",
                "sensor": sensor,
                "authenticity": authenticity,
                "fake_subcategory": fake_sub,
                "material": mat,
                "combo_key": ck,
                "path_hint": path_hint,
                "style_key": style_key,
            }
        )

    lora_meta = os.path.join(out_dir, "metadata_lora_all.jsonl")
    lora_idx = os.path.join(out_dir, "class_index_lora_all.json")
    combo_meta = os.path.join(out_dir, "metadata_prompt_combo.jsonl")
    combo_idx = os.path.join(out_dir, "class_index_prompt_combo.json")
    manifest_path = os.path.join(out_dir, "combo_manifest.json")

    write_jsonl(lora_meta, lora_records)
    write_class_index(
        lora_idx,
        root=root,
        variant="lora_all",
        style_to_id=OrderedDict([("livdet2015_all", 0)]),
        num_images=len(lora_records),
        corrupt_count=len(corrupt),
        id_to_init_prompt={0: lora_text},
    )
    write_jsonl(combo_meta, combo_records)
    write_class_index(
        combo_idx,
        root=root,
        variant="prompt_combo",
        style_to_id=combo_style_to_id,
        num_images=len(combo_records),
        corrupt_count=len(corrupt),
        id_to_init_prompt=combo_init,
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(combo_manifest, f, ensure_ascii=False, indent=2)

    bad_path = os.path.join(out_dir, "corrupt_images.jsonl")
    with open(bad_path, "w", encoding="utf-8") as f:
        for row in corrupt:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "root_dir": root,
        "out_dir": out_dir,
        "lora_images": len(lora_records),
        "combo_images": len(combo_records),
        "num_combo_classes": len(combo_style_to_id),
        "combo_keys": list(combo_style_to_id.keys()),
        "corrupt": len(corrupt),
    }
    summary_path = os.path.join(out_dir, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
