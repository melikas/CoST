"""Shared figure style for the ``scripts/`` plotting entry points.

Single source of the project palette and axis styling, imported by
``dataset_stats.py`` and ``results_figures.py`` so their two figure sets cannot
drift apart. Importing this selects the Agg backend, so import it before pyplot.

``C_HRD`` / ``C_GLB`` are categorical slots 1 and 2; the pair was validated for
colour-vision deficiency before use (worst-pair CVD dE 24.7, normal 33.6).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
POS, NEG, ACCENT = "#2a78d6", "#d03b3b", "#eb6834"
C_HRD, C_GLB, CRIT = POS, ACCENT, NEG          # per-dataset colours reuse the same slots

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "font.size": 9,
})


def strip(ax):
    """Drop the top/right/left spines and tick marks (keeps the baseline)."""
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def save(fig, out_dir, name, dpi=200):
    """Write ``fig`` to ``out_dir/name``, close it, and echo the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return path
