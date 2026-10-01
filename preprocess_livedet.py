#!/usr/bin/env python3
"""Extract square fingerprint ROIs while preserving the official dataset splits."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_dir", type=Path, required=True, help="Dataset root containing Training/ and Testing/.")
    parser.add_argument("--dst_dir", type=Path, required=True, help="New directory for ROI images with the same layout.")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--dry_run", action="store_true", help="List planned counts without writing files.")
    args = parser.parse_args()
    source = args.src_dir.resolve()
    destination = args.dst_dir.resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination directories must not overlap.")
    if args.patch_size < 1:
        raise ValueError("--patch_size must be positive.")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = []
    counts = {}
    for split in ("Training", "Testing"):
        split_root = source / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing official split: {split_root}")
        selected = [p for p in sorted(split_root.rglob("*")) if p.is_file() and p.suffix.lower() in extensions
                    and not any(x.startswith(("@", "._")) or x.lower() == "generated" for x in p.relative_to(source).parts)
                    and not p.name.startswith("SYNOFILE")]
        if not selected:
            raise ValueError(f"No images in {split_root}")
        counts[split] = len(selected)
        images.extend(selected)
    print(json.dumps({"images": counts, "resolution": args.patch_size}, indent=2))
    targets = [p.relative_to(source).with_suffix(".png") for p in images]
    if len(targets) != len(set(targets)):
        raise ValueError("Source filenames with different extensions share an output stem. Give those sources unique names first.")
    if args.dry_run:
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty to prevent mixing preprocessing runs.")
    import cv2
    from ROI_resize import extract_patch_from_image
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fingerprint_gen/sd15"))
    from utils.run_logger import save_json, sha256_file, software_versions
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "preprocessing_config.json",
              {"source_root": str(source), "patch_size": args.patch_size, "profile": "default",
               "software": software_versions(), "roi_sha256": sha256_file(Path(__file__).with_name("ROI_resize.py"))})
    with (destination / "preprocessing_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for i, path in enumerate(images, 1):
            if not path.resolve().is_relative_to(source):
                raise ValueError(f"Image symlink leaves dataset root: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            patch, scaled, padded = extract_patch_from_image(image, args.patch_size)
            if patch is None:
                raise ValueError(f"Cannot extract ROI: {path}")
            # Keep the source stem used by the generation seed rule.
            relative = path.relative_to(source)
            target = destination / relative.with_suffix(".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.stem + ".tmp.png")
            if not cv2.imwrite(str(temporary), patch):
                raise OSError(f"Could not write {temporary}")
            os.replace(temporary, target)
            stream.write(json.dumps({"source": relative.as_posix(), "source_sha256": sha256_file(path),
                                     "output": target.relative_to(destination).as_posix(),
                                     "output_sha256": sha256_file(target), "scaled": scaled, "padded": padded}) + "\n")
            stream.flush()
            if i % 500 == 0:
                print(f"Processed {i}/{len(images)}")
    print(f"Saved {len(images)} ROI images under {destination}")

if __name__ == "__main__":
    main()

