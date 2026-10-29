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

STREAMS = ["electricity", "rotating_hyperplane", "sea"]
NICE = {"electricity": "Electricity (real)", "rotating_hyperplane": "Rotating hyperplane (gradual)",
        "sea": "SEA (sudden)"}
# Every text uses one size, fitted so that it prints at the 7 pt floor when the
# figure is placed at the single-column text width, 390 TeX points (author
# request, 2026-10-27).
PRINT_WIDTH_IN = 390 / 72.27
PRINT_PT = 7.0


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
    png_path = fig_dir / "fig6_streaming.png"

    def draw(fs):
        plt.rcParams.update({
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
        })
        fig, axes = plt.subplots(2, 3, figsize=(9.3, 6.0))
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
            # Line style as well as hue, so the three series stay distinguishable
            # when the page is printed in grayscale.
            for lab, ser, c, ls in [("controller", wseries("ctrl", 0.0), "#c0392b", "-"),
                                    ("no decay", wseries("fixed", 0.0), "#444", "--"),
                                    (f"best fixed (lam={best_lam})", wseries("fixed", best_lam), "#2c7fb8", ":")]:
                sm = ser.rolling(3, min_periods=1).mean()
                ax.plot(sm.index, sm.values, label=lab, color=c, lw=1.4, ls=ls)
            ax.set_title(f"{NICE[s]}\n(best fixed $\\lambda$={best_lam:g})")
            ax.set_xlabel("stream window")
            # Each row shares one quantity, so only its first panel names it;
            # repeated labels ran into the neighbouring panel's title.
            if j == 0:
                ax.set_ylabel("prequential accuracy")
            ax.grid(alpha=0.25)

            axl = axes[1, j]
            lam_ser = sub[sub.method == "ctrl"].groupby("window")["lam"].mean()
            axl.plot(lam_ser.index, lam_ser.values, color="#c0392b", lw=1.4)
            axl.set_xlabel("stream window")
            if j == 0:
                axl.set_ylabel(r"controller $\lambda$")
            axl.grid(alpha=0.25)
        for index, ax in enumerate(axes.ravel()):
            # The distance below the axes grows with the tick and axis labels
            # it has to clear.
            ax.annotate(
                f"({chr(97 + index)})",
                xy=(0.5, 0),
                xycoords="axes fraction",
                xytext=(0, -43 * fs / 10.2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=fs,
                fontweight="normal",
                annotation_clip=False,
            )
        handles, _ = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            ["controller", "no decay", "best fixed"],
            fontsize=fs,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0)
        return fig

    def save(fig):
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

    # Redraw until the text prints at PRINT_PT: the printed size is the drawn
    # size times the text width over the width of the cropped PNG.
    fs = PRINT_PT
    for _ in range(6):
        save(draw(fs))
        with Image.open(png_path) as im:
            printed = fs * PRINT_WIDTH_IN * 600 / im.width
        if PRINT_PT <= printed < PRINT_PT + 0.05:
            print(f"fig6_streaming: drawn at {fs:.2f} pt, prints at {printed:.2f} pt")
            break
        fs *= (PRINT_PT + 0.02) / printed
    else:
        raise SystemExit(f"fig6_streaming: printed text did not settle at {PRINT_PT} pt ({printed:.3f})")
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
