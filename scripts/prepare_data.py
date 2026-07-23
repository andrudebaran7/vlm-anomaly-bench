#!/usr/bin/env python
"""Verify an extracted MVTec AD 2 tree before anything expensive runs on it.

A twelve-hour grid that dies on a missing mask has wasted a session. This fails in seconds
instead, and names the file.

Usage:
    python scripts/prepare_data.py --root data/mvtec_ad2
    python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError

from vlmab.datasets.mvtec_ad2 import MVTecAD2, lighting_condition

_EXPECTED_DIRS = {
    "train": ("good",),
    "validation": ("good",),
    "test_public": ("good", "bad"),
    "test_private": (),
    "test_private_mixed": (),
}


def _image_dirs(category_root: Path, split: str) -> list[Path]:
    subdirs = _EXPECTED_DIRS[split]
    return [category_root / split / s for s in subdirs] if subdirs else [category_root / split]


def check_category(root: Path, category: str) -> list[str]:
    """Problems found in one category. Empty list means the layout is sound."""
    category_root = Path(root) / category
    problems: list[str] = []

    for split in MVTecAD2.SPLITS:
        for directory in _image_dirs(category_root, split):
            if not directory.is_dir():
                problems.append(f"{category}/{split}: missing directory {directory}")
                continue
            images = sorted(directory.glob("*.png"))
            if not images:
                problems.append(f"{category}/{split}: {directory} is empty")
            for image in images:
                try:
                    lighting_condition(image)
                except ValueError as exc:
                    problems.append(f"{category}/{split}: {exc}")
                try:
                    with Image.open(image) as im:
                        im.verify()
                except (UnidentifiedImageError, OSError) as exc:
                    problems.append(f"{category}/{split}: {image} is not a readable PNG ({exc})")

    ground_truth = category_root / "test_public" / "ground_truth" / "bad"
    bad = category_root / "test_public" / "bad"
    if not ground_truth.is_dir():
        problems.append(f"{category}/test_public: missing directory {ground_truth}")
    elif bad.is_dir():
        bad_stems = {p.stem for p in bad.glob("*.png")}
        mask_stems = {p.stem[: -len("_mask")] for p in ground_truth.glob("*_mask.png")}
        for stem in sorted(bad_stems - mask_stems):
            problems.append(f"{category}/test_public: no mask for bad image {stem}.png")
        for stem in sorted(mask_stems - bad_stems):
            problems.append(f"{category}/test_public: orphan mask {stem}_mask.png has no image")

    return problems


def summarize_category(root: Path, category: str) -> dict[str, dict[str, int]]:
    """Per split, how many images of each lighting condition."""
    category_root = Path(root) / category
    summary: dict[str, dict[str, int]] = {}
    for split in MVTecAD2.SPLITS:
        counts: Counter[str] = Counter()
        for directory in _image_dirs(category_root, split):
            if not directory.is_dir():
                continue
            for image in directory.glob("*.png"):
                try:
                    counts[lighting_condition(image)] += 1
                except ValueError:
                    counts["<unparseable>"] += 1
        summary[split] = dict(sorted(counts.items()))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="dataset root directory")
    parser.add_argument("--category", help="check one category instead of all of them")
    args = parser.parse_args(argv)

    categories = [args.category] if args.category else MVTecAD2(args.root).categories()
    if not categories:
        print(f"no categories found under {args.root}")
        return 1

    failed = False
    for category in categories:
        problems = check_category(args.root, category)
        summary = summarize_category(args.root, category)
        total = sum(sum(c.values()) for c in summary.values())
        print(f"\n{category}: {total} images")
        for split, counts in summary.items():
            rendered = ", ".join(f"{k}={v}" for k, v in counts.items()) or "-"
            print(f"  {split:<20} {rendered}")
        for problem in problems:
            print(f"  PROBLEM: {problem}")
            failed = True

    print("\nFAILED" if failed else "\nOK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
