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
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
