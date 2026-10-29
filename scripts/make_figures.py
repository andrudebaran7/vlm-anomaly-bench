#!/usr/bin/env python3
"""Write the paper's map-free figure from the committed MVTec AD 2 reports.

    results/figures/anchor_per_condition.pdf

The paper copies results/figures/*.pdf in with its scripts/pull_results.sh. Needs matplotlib
(`pip install -e .[plots]`); nothing else here does.
"""
import argparse
from pathlib import Path

from vlmab.viz import figures, tables

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path,
                        default=ROOT / "results" / "mvtec_ad2" / "patchcore_ref_v0.2.21")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()

    reports = {c: (args.reports / f"{c}.md").read_text() for c in tables.CATEGORY_ORDER}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "anchor_per_condition.pdf"
    figures.draw_per_condition(reports, path)
    print(f"wrote {path.relative_to(ROOT)}")

    # The image figures need the pre-registered selections copied from Drive (notebook 7.2/7.3).
    # They show MVTec AD 2 images, so they are written to a gitignored folder and only ever reach
    # the paper, never this public repository.
    inputs = ROOT / "results" / "figure_inputs"
    if not inputs.is_dir():
        print(f"no {inputs.relative_to(ROOT)}: image figures skipped")
        return
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name, draw in (("dataset_lighting", figures.draw_dataset),
                       ("sheet_metal_resize", figures.draw_resize),
                       ("walnuts_vs_can", figures.draw_maps)):
        draw(inputs, images / f"{name}.pdf")
        print(f"wrote {(images / f'{name}.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
