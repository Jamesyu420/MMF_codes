"""Shared figure pieces for spatial programs."""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mmf_matplotlib")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"MMF": "#25627D", "RCTD": "#82AE9E", "STAMP": "#CC956B", "SpiceMix": "#9B8BB9", "NMF": "#A8AAAC"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "pdf.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False, "savefig.facecolor": "white",
})


def save(fig, path):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(path.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-.18, 1.13, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


def centered(v):
    v = v - v.mean()
    return v / max(float(np.sqrt(np.mean(v * v))), 1e-15)


def grid(coords, v):
    x = np.unique(coords[:, 0])
    y = np.unique(coords[:, 1])
    image = np.full((len(y), len(x)), np.nan)
    image[np.searchsorted(y, coords[:, 1]), np.searchsorted(x, coords[:, 0])] = v
    return image


def tissue(ax, coords, value, title, *, mask=None, lim=None, cmap="RdBu_r", size=1.6, invert=False):
    ax.scatter(coords[:, 0], coords[:, 1], c="#DFDFDF", s=max(size * .8, .3), linewidths=0, rasterized=True)
    shown = np.ones(len(coords), dtype=bool) if mask is None else mask
    image = ax.scatter(coords[shown, 0], coords[shown, 1], c=value[shown], s=size, linewidths=0,
                       rasterized=True, cmap=cmap, vmin=None if lim is None else lim[0],
                       vmax=None if lim is None else lim[1])
    ax.set_aspect("equal")
    if invert:
        ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(title, pad=4)
    return image


def gene_weights(ax, weights, genes, title, n=10, highlight=()):
    order = np.argsort(-weights, kind="stable")[:n][::-1]
    names = genes[order]
    ax.barh(range(n), weights[order], color=["#006F61" if g in highlight else "#82AE9E" for g in names], height=.72)
    ax.set_yticks(range(n), names, fontsize=6.5)
    ax.set_xlabel("Native gene weight")
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)


def symmetric_limit(values, mask, q=0.995):
    shown = np.abs(values[mask])
    return max(float(np.quantile(shown, q)), 1e-10)
