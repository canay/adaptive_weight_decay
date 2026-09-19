#!/usr/bin/env python3
"""Figure for the delayed-label streaming evaluation under concept drift.

Top row: windowed prequential accuracy (seed-mean, smoothed) for the controller,
no-decay, and the best fixed decay, per stream.
Bottom row: controller weight-decay coefficient over the stream (seed-mean).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageChops, ImageOps  # noqa: E402

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

STREAMS = ["electricity", "rotating_hyperplane", "sea"]
NICE = {"electricity": "Electricity (real)", "rotating_hyperplane": "Rotating hyperplane (gradual)",
        "sea": "SEA (sudden)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(d / "streaming_traces.csv")
    raw = pd.read_csv(d / "streaming_raw.csv")
    GRID = [0.0, 0.01, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.3))
    for j, s in enumerate(STREAMS):
        sub = tr[tr.stream == s]
        # best fixed (post-hoc oracle) by overall prequential acc
        fixed = {lam: raw[(raw.stream == s) & (raw.method == "fixed") & np.isclose(raw.lam0, lam)]["prequential_acc"].mean() for lam in GRID}
        best_lam = max(fixed, key=fixed.get)

        def wseries(method, lam0=None):
            q = sub[sub.method == method]
            if lam0 is not None:
                q = q[np.isclose(q.lam0, lam0)]
            return q.groupby("window")["acc"].mean()

        ax = axes[0, j]
        for lab, ser, c in [("controller", wseries("ctrl", 0.0), "#c0392b"),
                            ("no decay", wseries("fixed", 0.0), "#444"),
                            (f"best fixed (lam={best_lam})", wseries("fixed", best_lam), "#2c7fb8")]:
            sm = ser.rolling(3, min_periods=1).mean()
            ax.plot(sm.index, sm.values, label=lab, color=c, lw=1.4)
        ax.set_title(f"{NICE[s]}\n(best fixed $\\lambda$={best_lam:g})")
        ax.set_xlabel("stream window")
        ax.set_ylabel("prequential accuracy")
        ax.grid(alpha=0.25)

        axl = axes[1, j]
        lam = wseries("ctrl", 0.0)  # placeholder to get index
        lam_ser = sub[sub.method == "ctrl"].groupby("window")["lam"].mean()
        axl.plot(lam_ser.index, lam_ser.values, color="#c0392b", lw=1.4)
        axl.set_xlabel("stream window")
        axl.set_ylabel("controller lambda")
        axl.grid(alpha=0.25)
    for index, ax in enumerate(axes.ravel()):
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            clip_on=True,
        )
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["controller", "no decay", "best fixed"],
        fontsize=12,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5)
    png_path = fig_dir / "fig6_streaming.png"
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
    fig.savefig(fig_dir / "fig6_streaming.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"-> {fig_dir/'fig6_streaming.png'}")


if __name__ == "__main__":
    main()
