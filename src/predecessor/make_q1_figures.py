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
# Axis labels follow the manuscript's own spelling of each task name.
TASK_LABEL = {"adult": "Adult", "bank": "Bank", "digits": "Digits", "phoneme": "Phoneme",
              "spambase": "Spambase", "qsar": "QSAR", "waveform": "Waveform",
              "mfeat": "MFeat", "satimage": "Satimage"}
# One shape per non-controller series in the Pareto figure, so the points stay
# separable in grayscale and under deuteranopia.
PARETO_MARKERS = {"no decay": "o", "median fixed": "s", "AdaDecay": "^",
                  "validation tuning": "P", "oracle (non-deployable)": "D"}
# The review copy places the regime figure at the text width, 390 TeX points.
PRINT_WIDTH_IN = 390 / 72.27
PRINT_PT = 7.0


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
    # The x value counts training runs spent on decay selection. The oracle is
    # the best of the seven fixed-grid runs, so it spends seven; validation
    # tuning sweeps the grid on a held-out split and then retrains once, so it
    # spends eight (author decision, 2026-10-09).
    pts = [("no decay", 1, cell_pool(df, names, 0.0), "#7f8c8d"),
           ("median fixed", 1, pm("median"), "#7fcdbb"),
           ("AdaDecay", 1, pm("adadecay"), "#9b59b6"),
           ("rule B", 1, pm("ctrlB"), "#c0392b"),
           ("rule A", 1, pm("ctrlA"), "#e67e22"),
           ("validation tuning", 8, pm("valtuned"), "#2c7fb8"),
           ("oracle (non-deployable)", 7, pm("oracle"), "#34495e")]
    for lab, cost, acc, c in pts:
        # The two controllers keep the star marker and the larger size; the
        # test is on the label, so it is written against the labels in use.
        # Every other series gets its OWN shape: four identical circles were
        # indistinguishable in grayscale, where three of them sat at luminance
        # 131, 117 and 103 of 255.
        is_controller = lab in ("rule A", "rule B")
        mk = "*" if is_controller else PARETO_MARKERS.get(lab, "o")
        ax.scatter([cost], [acc], s=160 if is_controller else 90, color=c, marker=mk, zorder=3, label=lab)
    # The figure prints at the column width (about 0.70 of its drawn width), so
    # 10.6 pt ticks and legend print near 7.4 pt: every size is 1.4 pt below the
    # earlier set, one printed point (author request, 2026-09-30).
    ax.set_xlabel("training runs spent on decay selection (cost)", fontsize=11.1)
    ax.set_ylabel("pooled test accuracy (18 task variants)", fontsize=11.1)
    ax.set_xticks([1, 7, 8])
    ax.tick_params(axis="both", labelsize=10.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_q1_pareto")

    # ---- fig: complementary-signal regime structure ----
    clean = [f"{t}_full" for t in TAB]
    noisy = [f"{t}_n20" for t in TAB]

    # Every text uses one size, fitted so that it prints at the 7 pt floor at
    # the single-column text width (author request, 2026-10-27).
    def draw_regime(fs):
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3), sharey=True)
        handles = labels = None
        # Series and panel names follow the manuscript: rule A, rule B, corrupted.
        for ax, grp, title in [(axes[0], clean, "clean"), (axes[1], noisy, "corrupted")]:
            xs = np.arange(len(grp))
            w = 0.2
            for j, (k, c, lab, hatch) in enumerate([("oracle", "#2c7fb8", "oracle", ""),
                                                    ("ctrlA", "#e67e22", "rule A", "//"),
                                                    ("ctrlB", "#c0392b", "rule B", "xx"),
                                                    ("median", "#7fcdbb", "median", "..")]):
                ax.bar(xs + (j - 1.5) * w, [per[n][k] for n in grp], w, color=c, label=lab,
                       hatch=hatch, edgecolor="white", linewidth=0.3)
            ax.set_xticks(xs)
            # Anchored at the tick, so each slanted name ends at its own group;
            # centre-rotated names ran together ("QSARWaveform").
            ax.set_xticklabels([TASK_LABEL[n.replace("_full", "").replace("_n20", "")] for n in grp],
                               rotation=40, ha="right", rotation_mode="anchor", fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            ax.set_title(title, fontsize=fs)
            ax.grid(axis="y", alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
        for index, ax in enumerate(axes):
            # A fixed distance below the axes clears the longest slanted name; a
            # fraction of the axes height put "(a)" on top of "Spambase". The
            # distance grows with the slanted names it has to clear.
            ax.annotate(
                f"({chr(97 + index)})",
                xy=(0.50, 0.0),
                xycoords="axes fraction",
                xytext=(0, -43 * fs / 7.1),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=fs,
                fontweight="normal",
                annotation_clip=False,
            )
        axes[0].set_ylabel("test accuracy", fontsize=fs)
        # The legend stays above the panels. Moving it below, as in
        # fig4_signal_gap, was tried and MEASURED: every variant enlarged the
        # largest blank band in the asset (12.15 percent to 22.03 percent at best),
        # because this figure's panel labels then sat 0.40 axes heights below the
        # axes (they now sit a fixed distance below, under the slanted task names).
        fig.legend(handles, labels, fontsize=fs, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 0.99), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    # Redraw until the text prints at PRINT_PT: the printed size is the drawn
    # size times the text width over the width of the cropped PNG.
    fs = PRINT_PT
    for _ in range(6):
        save(draw_regime(fs), "fig_q1_regime")
        with Image.open(fd / "fig_q1_regime.png") as im:
            printed = fs * PRINT_WIDTH_IN * 600 / im.width
        if PRINT_PT <= printed < PRINT_PT + 0.05:
            print(f"fig_q1_regime: drawn at {fs:.2f} pt, prints at {printed:.2f} pt")
            break
        fs *= (PRINT_PT + 0.02) / printed
    else:
        raise SystemExit(f"fig_q1_regime: printed text did not settle at {PRINT_PT} pt ({printed:.3f})")
    print("pooled: oracle %.3f valtuned %.3f ctrlB %.3f ctrlA %.3f adaD %.3f median %.3f" %
          (pm("oracle"), pm("valtuned"), pm("ctrlB"), pm("ctrlA"), pm("adadecay"), pm("median")))
    print(f"figures -> {fd}")


def cell_pool(df, names, lam):
    import numpy as np
    return float(np.mean([df[(df.name == n) & (df.method == "fixed") & np.isclose(df.lam0, lam)]["test_acc"].mean()
                          for n in names]))


if __name__ == "__main__":
    main()
