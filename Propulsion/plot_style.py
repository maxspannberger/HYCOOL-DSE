"""
plot_style.py
=============
Centralised plotting aesthetic for the propulsion sizing chain.

Goal: every figure in the package looks like it belongs in the same report.
A single muted-cyan tonal palette is used everywhere; different curves on
the same plot are distinguished by tone (light -> dark) rather than hue.

Public API
----------
apply_style()
    Sets the global matplotlib rcParams. Call once before drawing.

CYAN_PALETTE
    Ordered list of hex colours, lightest to darkest. Use directly for
    line colours when a plot has a fixed number of series.

cyan_tones(n)
    Returns `n` evenly-spaced cyan tones. Useful when the number of series
    is computed at runtime.

CYAN_CMAP
    A sequential colormap matching the palette, for filled regions / heatmaps.

NEUTRAL_GREY / GRID_GREY / TEXT_GREY / ACCENT_WARN
    Off-palette colours for axes, gridlines, secondary annotations, and
    warnings (kept restrained so cyan stays dominant).

save_figure(fig, name, output_dir, dpi=200)
    Save a figure as a report-ready PNG and close it.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --- Tonal palette: muted cyan, lightest -> darkest ---------------------
CYAN_PALETTE = [
    "#cfe6ea",   # 0  whisper cyan (backgrounds, faint references)
    "#a8d3da",   # 1  pale
    "#7cbac4",   # 2  soft
    "#4fa0ad",   # 3  medium
    "#2f8590",   # 4  deep
    "#1f5c66",   # 5  shadow
    "#0e3438",   # 6  near-black cyan (headline curves)
]

# Neutrals -- restrained, off-palette
NEUTRAL_GREY = "#444444"
TEXT_GREY    = "#222222"
GRID_GREY    = "#dddddd"
LIGHT_BG     = "#fafbfc"

# Single warning accent (a desaturated rust). Used sparingly for limit lines.
ACCENT_WARN  = "#b76060"

# Sequential colormap built from the palette (for fills / heatmaps)
CYAN_CMAP = LinearSegmentedColormap.from_list("muted_cyan", CYAN_PALETTE, N=256)


# ---------------------------------------------------------------------------
# Tone selection
# ---------------------------------------------------------------------------
def cyan_tones(n, lightest=1, darkest=6):
    """
    Return `n` distinct cyan hex codes sampled across the palette.

    Sampling goes through the LinearSegmentedColormap built from the palette,
    so any `n` produces visually distinct tones (no duplicates when n exceeds
    the palette length). `lightest` and `darkest` are palette indices that
    bound the sampled range -- defaults stop short of pure-white / near-black
    so curves stay legible on a white background.
    """
    import numpy as np
    import matplotlib.colors as mcolors

    if n <= 0:
        return []
    L = len(CYAN_PALETTE) - 1
    if n == 1:
        t = 0.5 * (lightest + darkest) / L
        return [mcolors.to_hex(CYAN_CMAP(t))]
    ts = np.linspace(lightest / L, darkest / L, n)
    return [mcolors.to_hex(CYAN_CMAP(t)) for t in ts]


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
def apply_style():
    """
    Apply the global matplotlib style. Idempotent — safe to call multiple times.
    """
    mpl.rcParams.update({
        # Figure / canvas
        "figure.facecolor":  LIGHT_BG,
        "figure.dpi":        110,
        "savefig.facecolor": "white",
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.15,

        # Axes
        "axes.facecolor":   "white",
        "axes.edgecolor":   NEUTRAL_GREY,
        "axes.labelcolor":  TEXT_GREY,
        "axes.titlecolor":  TEXT_GREY,
        "axes.titleweight": "semibold",
        "axes.titlesize":   12.5,
        "axes.labelsize":   10.5,
        "axes.linewidth":   0.9,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "axes.axisbelow":   True,
        "axes.prop_cycle":  mpl.cycler(color=CYAN_PALETTE[1:6]),

        # Grid
        "grid.color":     GRID_GREY,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "grid.alpha":     0.85,

        # Ticks
        "xtick.color":     TEXT_GREY,
        "ytick.color":     TEXT_GREY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Lines / markers
        "lines.linewidth":   2.0,
        "lines.markersize":  4.5,
        "lines.solid_capstyle": "round",

        # Legend
        "legend.frameon":     False,
        "legend.fontsize":    9.0,
        "legend.labelcolor":  TEXT_GREY,
        "legend.borderpad":   0.4,

        # Font (keep system default; just size + family hints)
        "font.family":  ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size":    10.0,
    })


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_figure(fig, name, output_dir="plots", dpi=200, close=True):
    """
    Save `fig` as a report-ready PNG and (optionally) close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename stem (no extension). `.png` is appended.
    output_dir : str or Path
        Directory to write into. Created if missing.
    dpi : int
        Output resolution.
    close : bool
        If True, close `fig` after saving so subsequent plt.show() calls
        don't accidentally re-display it.

    Returns the absolute Path of the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path
