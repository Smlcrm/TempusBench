"""Render the synthetic taskbed overview figures embedded in synthetic_tasks.md.

Produces figures/synthetic_taskbed_full.png (all T=2048 points per task) and
figures/synthetic_taskbed_zoom.png (a 240-step window, t=1200..1440, chosen to
straddle the variance-shift breakpoint at 0.6T and reveal high-frequency
structure - seasonal shape, discreteness, lead-lag offsets - that the full
view compresses).  Regenerate after any generator change:

    python make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synthetic_generators as sg

DISCRETE = {"binary_latent_ar", "ordinal_categorical", "poisson_counts",
            "negbin_counts", "skellam_integer", "intermittent_demand",
            "intermittent_bursty", "lumpy_demand", "zero_inflated_continuous"}
MV_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def render(zoom, out, dpi=110):
    data = sg.generate_all(seed=0)
    names = list(data.keys())
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 2.1 * nrows))
    for ax in axes.flat:
        ax.set_visible(False)
    for i, name in enumerate(names):
        ax = axes.flat[i]
        ax.set_visible(True)
        arr = data[name]
        sl = slice(None) if zoom is None else slice(zoom[0], zoom[1])
        t = np.arange(arr.shape[0])[sl]
        cats = ",".join(sg.TASKS[name]["categories"][:3])
        lw = 0.4 if zoom is None else 0.9
        if arr.ndim == 1:
            if name in DISCRETE:
                ax.step(t, arr[sl], lw=max(lw, 0.5), color=MV_COLORS[0],
                        where="mid")
            else:
                ax.plot(t, arr[sl], lw=lw, color=MV_COLORS[0])
        else:
            for j in range(arr.shape[1]):
                ax.plot(t, arr[sl, j], lw=lw,
                        color=MV_COLORS[j % len(MV_COLORS)], alpha=0.75)
        ax.set_title(f"{name}\n[{cats}]", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("TempusBench synthetic taskbed (seed=0)" +
                 ("" if zoom is None else f"  -  zoom t={zoom[0]}..{zoom[1]}"),
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)
    render(None, os.path.join(figdir, "synthetic_taskbed_full.png"))
    render((1200, 1440), os.path.join(figdir, "synthetic_taskbed_zoom.png"))
