"""The paper's map-free figure: the anchor per lighting condition, from the committed reports.

Spec 2026-10-01 §5 lists it as the fifth figure ("per-condition metrics as small multiples,
straight from the shards"); it needs no anomaly map, so it is the one that can be drawn without a
Colab session. Drawn from the reports rather than the shards for the same reason as the tables:
the reports are the committed record.

Two panels side by side, one row per category in Table 2's order: I-AUROC per condition against
chance (0.5), and AU-PRO@0.05 per condition as a multiple of its random baseline (log scale,
against 1x). The training condition, `regular`, is the one highlighted: it is the only condition
the memory bank has seen, and §6.A reports that it is never the best one for detection.
Every other condition is gray. Two metrics on two scales are two panels, never two axes.
"""
from dataclasses import dataclass

from vlmab.viz import tables

#: The training condition, highlighted. Every MVTec AD 2 category trains on `regular` only.
TRAIN_CONDITION = "regular"

#: Emphasis palette: one hue for the highlighted condition, a neutral for the rest
#: (dataviz reference palette, light: categorical slot 1 and the muted axis gray).
HIGHLIGHT = "#2a78d6"
CONTEXT = "#898781"
INK = "#52514e"


@dataclass(frozen=True)
class Point:
    category: str
    condition: str
    i_auroc: float
    au_pro_005_x: float      # AU-PRO@0.05 as a multiple of the random baseline


def points(reports: dict[str, str]) -> list[Point]:
    """One point per (category, condition), Table 2's category order."""
    base = tables.RANDOM_BASELINE["au_pro_005"]
    out = []
    for c in tables.CATEGORY_ORDER:
        for r in tables.parse_report(reports[c]):
            out.append(Point(c, r.condition, float(r.mean["i_auroc"]),
                             float(r.mean["au_pro_005"] / base)))
    return out


def category_means(reports: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Each category's mean over conditions, as Table 2 reports it, on the figure's two axes."""
    base = tables.RANDOM_BASELINE["au_pro_005"]
    out = {}
    for c in tables.CATEGORY_ORDER:
        s = tables.summarise(tables.parse_report(reports[c]))
        out[c] = (float(s.i_auroc), float(s.au_pro_005 / base))
    return out


def draw_per_condition(reports: dict[str, str], path) -> None:
    """Write the figure to `path` (PDF for the paper)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # STIX: Times-like, shipped with matplotlib as TrueType, so it embeds cleanly as Type 42
        # (the system Nimbus Roman is OpenType CFF and embeds with a type mismatch). Mathtext uses
        # it too, so the multiplication sign is not a stray DejaVu glyph.
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    pts = points(reports)
    means = category_means(reports)
    cats = list(tables.CATEGORY_ORDER)
    row = {c: i for i, c in enumerate(cats)}
    n_usable = sum(1 for c in cats if c in tables.USABLE)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = (
        (axes[0], "i_auroc", 0.5, "chance", "I-AUROC per lighting condition", False),
        (axes[1], "au_pro_005_x", 1.0, "random", r"AU-PRO@0.05 per condition, $\times$ random baseline", True),
    )
    for ax, field, ref, ref_label, xlabel, log in panels:
        if log:
            ax.set_xscale("log")
        ax.axvline(ref, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.axhline(n_usable - 0.5, color=CONTEXT, lw=0.5, zorder=1)
        for p in pts:
            y = row[p.category]
            x = getattr(p, field)
            if p.condition == TRAIN_CONDITION:
                ax.scatter(x, y, marker="D", s=26, color=HIGHLIGHT, edgecolor="white",
                           linewidth=0.8, zorder=4)
            else:
                ax.scatter(x, y, marker="o", s=18, facecolor="none", edgecolor=CONTEXT,
                           linewidth=0.9, zorder=3)
        for c, (mi, mx) in means.items():
            x = mi if field == "i_auroc" else mx
            ax.plot([x, x], [row[c] - 0.32, row[c] + 0.32], color="black", lw=1.4, zorder=5)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e4e3df", lw=0.5, zorder=0)
        # The reference line would sit on a grid line of the same x; keep only the reference.
        for line in ax.get_xgridlines():
            if abs(line.get_xdata()[0] - ref) < 1e-9:
                line.set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(ref, -0.75, f" {ref_label}", color=INK, fontsize=7, va="center", ha="left")

    axes[0].set_xlim(0.3, 1.0)
    axes[1].set_xlim(0.15, 40)
    axes[1].set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
    axes[1].set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    axes[0].set_yticks(range(len(cats)))
    axes[0].set_yticklabels([tables.NAMES[c] for c in cats])
    axes[0].set_ylim(len(cats) - 0.5, -1.1)
    axes[0].text(0.305, n_usable - 0.62, "usable", color=INK, fontsize=7, va="bottom")
    axes[0].text(0.305, n_usable - 0.38, "not usable", color=INK, fontsize=7, va="top")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="D", ls="", color=HIGHLIGHT, markeredgecolor="white", markersize=5,
               label="regular (the training condition)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=CONTEXT,
               markersize=4.5, label="other lighting conditions"),
        Line2D([], [], color="black", lw=1.4, label="category mean, as in the anchor table"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.02))
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# --- Image figures, from the pre-registered selections (docs/next-steps.md, 2026-10-29) --------
#
# Inputs are `results/figure_inputs/`, copied from Drive after notebook 7.2/7.3; they hold
# MVTec AD 2 images and are gitignored (the dataset is not redistributed in this repository), and
# so are the PDFs these functions write (`results/figures/images/`).

#: Sequential single-hue ramp for anomaly maps (reference palette, blue 100 -> 700): light is a
#: low score, dark a high one.
MAP_RAMP = ("#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
#: The ground-truth outline drawn on a map has a different job from the map, so a different hue
#: (categorical slot 2).
OUTLINE = "#eb6834"
CONDITIONS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3")


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "pdf.fonttype": 42, "text.color": INK,
    })
    return plt


def _thumb(path, width):
    """The image downsampled for print, so a PDF does not carry five-megapixel frames."""
    from PIL import Image
    import numpy as np

    im = Image.open(path).convert("RGB")
    h = round(im.height * width / im.width)
    return np.asarray(im.resize((width, h), Image.Resampling.LANCZOS))


def _no_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def draw_dataset(inputs, path) -> None:
    """One Walnuts scene under each lighting condition, then an anomalous image and its mask."""
    from pathlib import Path

    plt = _style()
    d = Path(inputs) / "walnuts_dataset"
    panels = [(d / f"good_{c}.png", c.replace("_", " ")) for c in CONDITIONS]
    panels += [(d / "bad_regular.png", "anomalous"),
               (d / "bad_regular_mask.png", "ground truth")]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 1.02), gridspec_kw={"wspace": 0.04})
    for ax, (p, label) in zip(axes, panels):
        ax.imshow(_thumb(p, 360), interpolation="none")
        ax.set_title(label, fontsize=7, pad=2)
        _no_axes(ax)
    fig.savefig(path, bbox_inches="tight", dpi=300, metadata={"CreationDate": None})
    plt.close(fig)


def draw_resize(inputs, path) -> None:
    """Sheet Metal's frame at its own aspect beside the anchor pre-processor's square resize."""
    from pathlib import Path

    plt = _style()
    d = Path(inputs) / "sheet_metal_resize"
    fig = plt.figure(figsize=(3.45, 1.55))
    wide = fig.add_axes([0.0, 0.42, 1.0, 0.25])        # 4:1, at true aspect
    square = fig.add_axes([0.355, -0.33, 0.29, 0.65])   # 1:1
    wide.imshow(_thumb(d / "good_regular.png", 1200), interpolation="none", aspect="equal")
    square.imshow(_thumb(d / "good_regular_resized.png", 256), interpolation="none",
                  aspect="equal")
    wide.set_title("Sheet Metal frame, 4224 $\\times$ 1056 (4:1)", fontsize=7, pad=2)
    square.set_title("after the anchor's resize, 256 $\\times$ 256", fontsize=7, pad=2)
    for ax in (wide, square):
        _no_axes(ax)
    fig.savefig(path, bbox_inches="tight", dpi=300, metadata={"CreationDate": None})
    plt.close(fig)


def draw_maps(inputs, path) -> None:
    """Walnuts against Can: image, ground truth, and the anchor's map with the truth outlined."""
    import json
    from pathlib import Path

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    plt = _style()
    cmap = LinearSegmentedColormap.from_list("blue_sequential", MAP_RAMP)
    rows = (("walnuts", "Walnuts"), ("can", "Can"))
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.35),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.22})
    for r, (cat, name) in enumerate(rows):
        d = Path(inputs) / cat
        sel = json.loads((d / "selection.json").read_text())
        mask_full = np.asarray(Image.open(d / "mask.png").convert("L")) > 0
        amap_full = np.load(d / "map.npy").astype(np.float32)
        width = 700
        h = round(mask_full.shape[0] * width / mask_full.shape[1])
        mask = np.asarray(Image.fromarray(mask_full.astype(np.uint8) * 255).resize(
            (width, h), Image.Resampling.NEAREST)) > 0
        amap = np.asarray(Image.fromarray(amap_full).resize((width, h), Image.Resampling.BILINEAR))
        axes[r, 0].imshow(_thumb(d / "image.png", width), interpolation="none")
        axes[r, 1].imshow(mask, cmap="gray", interpolation="none")
        im = axes[r, 2].imshow(amap, cmap=cmap, interpolation="none")
        axes[r, 2].contour(mask, levels=[0.5], colors=[OUTLINE], linewidths=0.9)
        # A defect of a few pixels is invisible at print size, so ring it, on the truth and on
        # the map, wherever it covers under 0.5% of the frame. The ring marks the place; the
        # outline inside it is still the truth.
        if mask_full.mean() < 0.005:
            from matplotlib.patches import Circle

            ys, xs = np.nonzero(mask)
            if ys.size == 0:   # vanished in the downsampling: locate it at full resolution
                ys, xs = np.nonzero(mask_full)
                ys, xs = ys * h / mask_full.shape[0], xs * width / mask_full.shape[1]
            for ax in (axes[r, 1], axes[r, 2]):
                ax.add_patch(Circle((xs.mean(), ys.mean()), radius=0.06 * width, fill=False,
                                    edgecolor=OUTLINE, linewidth=1.0, linestyle=(0, (3, 1.5))))
        cb = fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("anomaly score", fontsize=6)
        axes[r, 0].set_ylabel(name, fontsize=8)
        ratio = sel["inside_outside_ratio"]
        axes[r, 2].set_title(f"anomaly map; inside / outside the truth: {ratio:.2f}", fontsize=7,
                             pad=2)
        axes[r, 0].set_title(f"anomalous image ({sel['rank'].split(' anomalous')[0]})",
                             fontsize=7, pad=2)
        axes[r, 1].set_title("ground truth", fontsize=7, pad=2)
        for ax in axes[r]:
            _no_axes(ax)
    fig.savefig(path, bbox_inches="tight", dpi=300, metadata={"CreationDate": None})
    plt.close(fig)
