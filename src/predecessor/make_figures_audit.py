#!/usr/bin/env python3
"""Rebuild manuscript figures from the clean audit rerun.

fig1_sensitivity : final test accuracy vs fixed lambda and vs controller start, 6 panels
fig2_lambda_traj : controller lambda trajectories on the noisy variants, all 7 starts
fig3_bars        : oracle / median / controller bars for accuracy, gap, ECE
fig4_signal_gap  : control signal and train-test gap, fixed lambda=0 vs controller
fig5_baselines   : controller vs AdaDecay / AdamP / cosine / step (accuracy), 6 panels

Outputs PNG (600 dpi) and vector PDF into the chosen figure directory.
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

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
TASKS = ["adult_full", "bank_full", "digits_full", "adult_n20", "bank_n20", "digits_n20"]
NICE = {"adult_full": "Adult", "bank_full": "Bank", "digits_full": "Digits",
        "adult_n20": "Adult-S", "bank_n20": "Bank-S", "digits_n20": "Digits-S"}
NOISY = ["adult_n20", "bank_n20", "digits_n20"]
BASELINES = ["adadecay_default", "adadecay_regime", "adamp_proj", "cosine_wd", "step_early_wd"]
BNICE = {"adadecay_default": "AdaDecay", "adadecay_regime": "AdaDecay(λ=1)",
         "adamp_proj": "AdamP", "cosine_wd": "cosine", "step_early_wd": "early/step"}
# Grayscale-safe encoding for the seven controller starts in fig2.
TRAJ_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
TRAJ_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def xpos(lams):
    # map lambda values to evenly spaced x positions (0 included)
    return list(range(len(lams)))


def grid_label(lam: float) -> str:
    """Write a grid value as the manuscript does: {0, 0.01, 0.1, 1, 3, 10, 30}."""
    return f"{lam:g}"


def add_panel_labels(axes, offset_pt: float | None = None) -> None:
    """Add publication-style panel identifiers without covering plotted data.

    With offset_pt the label sits a fixed distance below the axes, so it clears
    the tick labels and the axis label whatever height tight_layout gives the
    axes; a fraction of the axes height moves when the layout shrinks the axes.
    """
    for index, ax in enumerate(np.asarray(axes).ravel()):
        if offset_pt is not None:
            ax.annotate(
                f"({chr(97 + index)})",
                xy=(0.50, 0.0),
                xycoords="axes fraction",
                xytext=(0, -offset_pt),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="normal",
                annotation_clip=False,
            )
            continue
        ax.text(
            0.50,
            -0.30,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="normal",
            clip_on=False,
        )


def crop_white_border(path: Path, padding_px: int = 12) -> None:
    """Crop renderer-added whitespace and retain a 0.02-inch safety edge."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    rgb = canvas.convert("RGB")
    bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
    if bbox is not None:
        rgb = rgb.crop(bbox)
    rgb = ImageOps.expand(rgb, border=padding_px, fill="white")
    rgb.save(path, dpi=(600, 600))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(d / "raw_runs.csv")
    if "diverged" in raw:
        raw = raw[~raw["diverged"].fillna(False).astype(bool)]
    tr = pd.read_csv(d / "controller_traces.csv")
    lr0 = 3e-3

    def save(fig, name):
        png_path = fig_dir / f"{name}.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        crop_white_border(png_path)
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # ---- fig1 sensitivity ----
    # Drawn at the printed width (the full text width, about 6.84 in), so every
    # point size below prints as set and no text falls under the 7 pt floor.
    # The two series are the same in all six panels, so one shared legend and
    # one shared axis label replace the six per-panel copies.
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 5.0))
    for ax, task in zip(axes.ravel(), TASKS):
        sub = raw[(raw.task == task) & (raw.lr == lr0)]
        fx = [sub[(sub.method == "fixed") & np.isclose(sub.lam0, l)]["test_acc"].mean() for l in GRID]
        ct = [sub[(sub.method == "ctrl") & np.isclose(sub.lam0, l)]["test_acc"].mean() for l in GRID]
        x = xpos(GRID)
        ax.plot(x, fx, "o-", label="fixed decay", color="#444", markersize=4, linewidth=1.2)
        ax.plot(x, ct, "s--", label="controller start", color="#c0392b", markersize=4, linewidth=1.2)
        ax.set_title(NICE[task], fontsize=10)
        ax.set_xticks(x)
        # Seven labels share about 1.6 in, so level labels touched ("0.00.010.1");
        # a 45-degree slant anchored at the tick keeps each one readable.
        ax.set_xticklabels([grid_label(l) for l in GRID], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9.5, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    # 11.5 pt keeps the lambda_0 subscript (0.7 of the label) above 8 pt in print.
    fig.supxlabel(r"$\lambda$ (fixed) or $\lambda_0$ (controller)", fontsize=11.5)
    fig.supylabel("final test accuracy", fontsize=10.5)
    add_panel_labels(axes, offset_pt=31)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.8, w_pad=0.8)
    save(fig, "fig1_sensitivity")

    # ---- fig2 lambda trajectories on noisy tasks, all starts ----
    # Drawn at the printed width (about 6.84 in); the axis label and legend
    # carry subscripts, so they stay at 11.5 pt to keep the subscript above 8 pt.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.1), sharey=True)
    for ax, task in zip(axes, NOISY):
        sub = tr[(tr.task == task) & (tr.method == "ctrl") & (tr.lr == lr0)]
        for idx, lam0 in enumerate(GRID):
            s = sub[np.isclose(sub.lam0.fillna(-1), lam0)]
            if not len(s):
                continue
            m = s.groupby("ep")["lam"].mean()
            # Line style and marker as well as hue, so the seven starts stay
            # distinguishable when the page is printed in grayscale.
            ax.plot(m.index, m.values, label=f"$\\lambda_0$={grid_label(lam0)}",
                    ls=TRAJ_STYLES[idx % len(TRAJ_STYLES)],
                    marker=TRAJ_MARKERS[idx % len(TRAJ_MARKERS)],
                    markevery=(idx, 7), markersize=3.2, linewidth=1.1)
        ax.set_title(NICE[task], fontsize=10)
        ax.set_xlabel("epoch", fontsize=10)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"controller $\lambda_t$", fontsize=11.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=11.5,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    # Below the "epoch" label, not at a fraction of the short axes height,
    # where "(a)" printed on top of "epoch".
    add_panel_labels(axes, offset_pt=34)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=0.6)
    save(fig, "fig2_lambda_traj")

    # ---- fig3 bars: oracle/median/controller for acc, gap, ece ----
    # recompute references per task
    import json
    stats = json.loads((d / "stats_audit.json").read_text()) if (d / "stats_audit.json").exists() else None
    metrics = [("acc", "test accuracy"), ("gap", "train-test gap"), ("ece", "ECE")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2))
    width = 0.26
    xs = np.arange(len(TASKS))
    for ax, (mk, mlabel) in zip(axes, metrics):
        orc, med, ctl = [], [], []
        for task in TASKS:
            r = stats["tasks"][task]
            orc.append(r[f"oracle_{mk}_mean"])
            med.append(r[f"median_{mk}_mean"])
            ctl.append(r[f"ctrl_{mk}_mean"])
        ax.bar(xs - width, orc, width, label="oracle fixed", color="#2c7fb8")
        ax.bar(xs, med, width, label="median fixed", color="#7fcdbb")
        ax.bar(xs + width, ctl, width, label="controller", color="#c0392b")
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[t] for t in TASKS], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(mlabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    add_panel_labels(axes)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig3_bars")

    # ---- fig4 signal vs gap: fixed lambda=0 vs controller on Adult-S and Digits-S ----
    # Drawn at the printed width (about 6.84 in). The right-axis label and one
    # legend entry carry a subscript, so they stay at 11 pt (subscript > 7 pt).
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3))
    legend_handles, legend_labels = [], []
    for ax, task in zip(axes, ["adult_n20", "digits_n20"]):
        f0 = tr[(tr.task == task) & (tr.method == "fixed") & np.isclose(tr.lam0.fillna(-1), 0.0) & (tr.lr == lr0)]
        c0 = tr[(tr.task == task) & (tr.method == "ctrl") & np.isclose(tr.lam0.fillna(-1), 0.0) & (tr.lr == lr0)]
        fg = f0.groupby("ep")["gap"].mean()
        cg = c0.groupby("ep")["gap"].mean()
        ax.plot(fg.index, fg.values, "-", label="gap (fixed $\\lambda$=0)", color="#444", linewidth=1.2)
        ax.plot(cg.index, cg.values, "--", label="gap (controller)", color="#c0392b", linewidth=1.2)
        if "s" in c0:
            cs = c0.groupby("ep")["s"].mean()
            ax2 = ax.twinx()
            ax2.plot(cs.index, cs.values, ":", color="#2c7fb8", label="signal $s_t$ (ctrl)", linewidth=1.3)
            ax2.set_ylabel("control signal $s_t$", fontsize=11)
            ax2.tick_params(axis="y", labelsize=8.5)
        ax.set_title(NICE[task], fontsize=10)
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("train-test gap", fontsize=10)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(alpha=0.25)
        if not legend_handles:
            handles, labels = ax.get_legend_handles_labels()
            if "s" in c0:
                signal_handles, signal_labels = ax2.get_legend_handles_labels()
                handles += signal_handles
                labels += signal_labels
            legend_handles, legend_labels = handles, labels
    add_panel_labels(axes)
    fig.legend(legend_handles, legend_labels, fontsize=11, loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    save(fig, "fig4_signal_gap")

    # ---- fig5 baselines vs controller (accuracy) ----
    fig, ax = plt.subplots(figsize=(12, 4.2))
    methods = ["ctrl"] + BASELINES
    labels = ["controller"] + [BNICE[b] for b in BASELINES]
    width = 0.8 / len(methods)
    xs = np.arange(len(TASKS))
    colors = ["#c0392b", "#2c7fb8", "#9b59b6", "#16a085", "#f39c12", "#7f8c8d"]
    for j, (mth, lab) in enumerate(zip(methods, labels)):
        vals = []
        for task in TASKS:
            sub = raw[(raw.task == task) & (raw.lr == lr0) & (raw.method == mth)]
            if mth == "ctrl":
                sub = sub[np.isclose(sub.lam0.fillna(-1), 0.0)]
            vals.append(sub["test_acc"].mean())
        ax.bar(xs + (j - len(methods) / 2) * width, vals, width, label=lab, color=colors[j % len(colors)])
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[t] for t in TASKS], rotation=20, ha="right")
    ax.set_ylabel("final test accuracy")
    ax.set_title("Controller vs close-prior baselines (validation-free, default settings)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save(fig, "fig5_baselines")

    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
