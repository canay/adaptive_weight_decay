#!/usr/bin/env python3
"""Q1 figures: (1) accuracy-vs-tuning-cost Pareto frontier, (2) complementary-signal
regime structure across datasets, (3) breadth comparison."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import numpy as np, pandas as pd  # noqa
from PIL import Image, ImageChops, ImageOps  # noqa

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
TAB = ["adult", "bank", "digits", "phoneme", "spambase", "qsar", "waveform", "mfeat", "satimage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    fd = Path(a.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(d / "q1_raw.csv")
    df = df[~df["diverged"].fillna(False)]
    names = [f"{t}_{v}" for v in ("full", "n20") for t in TAB]

    def cell(name, method, lam=None):
        s = df[(df.name == name) & (df.method == method)]
        if lam is not None:
            s = s[np.isclose(s.lam0, lam)]
        return float(s["test_acc"].mean())

    per = {n: dict(oracle=max(cell(n, "fixed", l) for l in GRID),
                   median=float(np.median([cell(n, "fixed", l) for l in GRID])),
                   ctrlA=cell(n, "ctrlA"), ctrlB=cell(n, "ctrlB"),
                   adadecay=cell(n, "adadecay_regime"), valtuned=cell(n, "valtuned")) for n in names}
    pm = lambda k: float(np.mean([per[n][k] for n in names]))

    def save(fig, name):
        png_path = fd / f"{name}.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        with Image.open(png_path) as source:
            rgba = source.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        rgb = canvas.convert("RGB")
        bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        if bbox is not None:
            rgb = rgb.crop(bbox)
        ImageOps.expand(rgb, border=12, fill="white").save(png_path, dpi=(600, 600))
        fig.savefig(fd / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # ---- fig: Pareto accuracy vs tuning cost ----
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    pts = [("no decay", 1, cell_pool(df, names, 0.0), "#7f8c8d"),
           ("median fixed", 1, pm("median"), "#7fcdbb"),
           ("AdaDecay", 1, pm("adadecay"), "#9b59b6"),
           ("controller B", 1, pm("ctrlB"), "#c0392b"),
           ("controller A", 1, pm("ctrlA"), "#e67e22"),
           ("validation tuning", 8, pm("valtuned"), "#2c7fb8"),
           ("oracle (non-deployable)", 8, pm("oracle"), "#34495e")]
    for lab, cost, acc, c in pts:
        mk = "*" if "controller" in lab else ("D" if "oracle" in lab else "o")
        ax.scatter([cost], [acc], s=160 if "controller" in lab else 90, color=c, marker=mk, zorder=3, label=lab)
    ax.set_xlabel("training runs spent on decay selection (cost)", fontsize=11.5)
    ax.set_ylabel("pooled test accuracy (18 task variants)", fontsize=11.5)
    ax.set_xticks([1, 8])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_q1_pareto")

    # ---- fig: complementary-signal regime structure ----
    clean = [f"{t}_full" for t in TAB]
    noisy = [f"{t}_n20" for t in TAB]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    handles = labels = None
    for ax, grp, title in [(axes[0], clean, "clean"), (axes[1], noisy, "noisy (label-corrupted)")]:
        xs = np.arange(len(grp))
        w = 0.2
        for j, (k, c, lab) in enumerate([("oracle", "#2c7fb8", "oracle"),
                                          ("ctrlA", "#e67e22", "ctrl A (norm-growth)"),
                                          ("ctrlB", "#c0392b", "ctrl B (loss-progress)"),
                                          ("median", "#7fcdbb", "median")]):
            ax.bar(xs + (j - 1.5) * w, [per[n][k] for n in grp], w, color=c, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("_full", "").replace("_n20", "") for n in grp], rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
    for index, ax in enumerate(axes):
        ax.text(
            0.50,
            -0.40,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="normal",
            clip_on=False,
        )
    axes[0].set_ylabel("test accuracy")
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig_q1_regime")
    print("pooled: oracle %.3f valtuned %.3f ctrlB %.3f ctrlA %.3f adaD %.3f median %.3f" %
          (pm("oracle"), pm("valtuned"), pm("ctrlB"), pm("ctrlA"), pm("adadecay"), pm("median")))
    print(f"figures -> {fd}")


def cell_pool(df, names, lam):
    import numpy as np
    return float(np.mean([df[(df.name == n) & (df.method == "fixed") & np.isclose(df.lam0, lam)]["test_acc"].mean()
                          for n in names]))


if __name__ == "__main__":
    main()
