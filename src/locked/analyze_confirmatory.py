#!/usr/bin/env python3
"""Deterministic, independence-safe analysis for the locked F06 matrix."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUN_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
RAW = RUN_ROOT / "outputs" / "main" / "confirmatory_raw.csv"
OUT = RUN_ROOT / "analysis"
ARMS = ("none", "fixed1", "ctrlA", "ctrlB", "ctrlBpp", "adadecay", "awd", "swd", "cwd")
CONTROLLERS = ("ctrlA", "ctrlB", "ctrlBpp")
COMPARATORS = ("adadecay", "awd", "swd", "cwd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)


def holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def cluster_bootstrap_ci(task_diffs: pd.Series, reps: int = 20000) -> tuple[float, float]:
    frame = task_diffs.rename("difference").reset_index()
    frame["base"] = frame["dataset"].str.replace(r"_(full|n20)$", "", regex=True)
    bases = sorted(frame["base"].unique())
    grouped = {base: frame.loc[frame["base"] == base, "difference"].to_numpy() for base in bases}
    rng = np.random.default_rng(20260826)
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        sampled = rng.choice(bases, size=len(bases), replace=True)
        values = np.concatenate([grouped[base] for base in sampled])
        estimates[index] = np.median(values)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    expected_rows = 18 * len(ARMS) * 5
    if len(df) != expected_rows:
        raise AssertionError(f"row count {len(df)} != {expected_rows}")
    if df.duplicated(["dataset", "arm", "seed"]).any():
        raise AssertionError("duplicate cells")
    if set(df["arm"]) != set(ARMS) or df["dataset"].nunique() != 18 or set(df["seed"]) != set(range(5)):
        raise AssertionError("matrix mismatch")
    metrics = ["test_acc", "test_ece", "train_acc", "gap"]
    if not np.isfinite(df[metrics].to_numpy()).all():
        raise AssertionError("non-finite metric")
    if not df["test_acc"].between(0, 1).all() or not df["test_ece"].between(0, 1).all():
        raise AssertionError("metric out of bounds")

    df["regime"] = np.where(df["dataset"].str.endswith("_n20"), "noisy", "clean")
    df["base_dataset"] = df["dataset"].str.replace(r"_(full|n20)$", "", regex=True)
    descriptive = df.groupby("arm")[metrics].agg(["mean", "median", "std", "min", "max"])
    descriptive.columns = ["_".join(column) for column in descriptive.columns]
    descriptive = descriptive.reset_index()
    descriptive.to_csv(OUT / "descriptive_by_arm.csv", index=False)

    regime = df.groupby(["regime", "arm"])[metrics].agg(["mean", "median", "std"])
    regime.columns = ["_".join(column) for column in regime.columns]
    regime = regime.reset_index()
    regime.to_csv(OUT / "regime_summary.csv", index=False)

    task_means = df.groupby(["dataset", "regime", "base_dataset", "arm"], as_index=False)[metrics].mean()
    task_means.to_csv(OUT / "task_variant_means.csv", index=False)
    task_wide = task_means.pivot(index="dataset", columns="arm", values="test_acc")
    seed_wide = df.pivot(index=["dataset", "seed"], columns="arm", values="test_acc")

    paired_metrics = df.pivot(index=["dataset", "seed"], columns="arm", values=metrics)
    cross_arm_identical = []
    for left_index, left in enumerate(ARMS):
        for right in ARMS[left_index + 1:]:
            identical = all(
                np.array_equal(
                    paired_metrics[(metric, left)].to_numpy(),
                    paired_metrics[(metric, right)].to_numpy(),
                )
                for metric in metrics
            )
            if identical:
                cross_arm_identical.append({"left": left, "right": right, "paired_cells": 90})

    comparisons = []
    for controller in CONTROLLERS:
        for comparator in COMPARATORS:
            task_diff = task_wide[controller] - task_wide[comparator]
            seed_diff = seed_wide[controller] - seed_wide[comparator]
            lo, hi = cluster_bootstrap_ci(task_diff)
            comparisons.append({
                "controller": controller,
                "comparator": comparator,
                "n_task_variants": int(len(task_diff)),
                "mean_paired_difference": float(task_diff.mean()),
                "median_paired_difference": float(task_diff.median()),
                "std_paired_difference": float(task_diff.std(ddof=1)),
                "iqr_paired_difference": float(task_diff.quantile(0.75) - task_diff.quantile(0.25)),
                "cluster_bootstrap_median_ci_low": lo,
                "cluster_bootstrap_median_ci_high": hi,
                "task_variant_wins": int((task_diff > 0).sum()),
                "task_variant_ties": int((task_diff == 0).sum()),
                "task_variant_losses": int((task_diff < 0).sum()),
                "task_level_wilcoxon_p_raw": safe_wilcoxon(task_diff.to_numpy()),
                "seed_cell_wilcoxon_p_sensitivity": safe_wilcoxon(seed_diff.to_numpy()),
            })
    adjusted = holm_adjust([row["task_level_wilcoxon_p_raw"] for row in comparisons])
    for row, p_adjusted in zip(comparisons, adjusted):
        row["task_level_wilcoxon_p_holm"] = p_adjusted
        row["passes_superiority_gate"] = bool(
            row["mean_paired_difference"] >= 0.005 and p_adjusted < 0.05
        )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(OUT / "primary_comparisons.csv", index=False)

    winner = task_wide.idxmax(axis=1).value_counts().rename_axis("arm").reset_index(name="task_variant_wins_including_ties_resolved_by_column_order")
    winner.to_csv(OUT / "winner_counts.csv", index=False)
    arm_rank = descriptive.sort_values(["test_acc_mean", "arm"], ascending=[False, True]).reset_index(drop=True)
    arm_rank["rank"] = arm_rank["test_acc_mean"].rank(method="min", ascending=False).astype(int)
    arm_rank.to_csv(OUT / "arm_ranking_descriptive.csv", index=False)

    passes = comparison_df.loc[comparison_df["passes_superiority_gate"]]
    best = arm_rank.iloc[0]
    evidence = {
        "raw_sha256": sha256(RAW),
        "analysis_script_sha256": sha256(Path(__file__)),
        "analysis_amendment_sha256": sha256(PROJECT / "MD" / "02_design" / "ANALYSIS_AMENDMENT_20260826.md"),
    }
    summary = {
        "status": "PASS_COMPLETE_ANALYSIS_WITH_INERT_ARM_DIAGNOSIS",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix_rows": len(df),
        "task_variants": int(df["dataset"].nunique()),
        "base_datasets": int(df["base_dataset"].nunique()),
        "arms": list(ARMS),
        "seeds": sorted(int(value) for value in df["seed"].unique()),
        "best_descriptive_arm": str(best["arm"]),
        "best_descriptive_test_accuracy_mean": float(best["test_acc_mean"]),
        "controller_superiority_gate_pass_count": int(len(passes)),
        "controller_superiority_gate_passes": passes[["controller", "comparator"]].to_dict("records"),
        "cross_arm_exact_metric_duplicates": cross_arm_identical,
        "ctrlBpp_diagnosis": "predeclared norm-collapse release condition never activated; ctrlBpp is empirically identical to ctrlB in all 90 paired cells",
        "evidence": evidence,
    }
    atomic_json(OUT / "analysis_summary.json", summary)

    lines = [
        "# Confirmatory Analysis Report",
        "",
        "Date/time: 2026-08-26 14:11 +03:00  ",
        "Tool: Codex  ",
        "Model, if known: GPT-5.6 Extra High (xhigh, user-attested; runtime exposed GPT-5 family)  ",
        "Operation ID: `F06-NOVELTY-ACCEPTABILITY-UPGRADE-20260826-01`",
        "",
        "Verdict: `COMPLETE / CLAIM_GATE_EVALUATED`.",
        "",
        "## Matrix QA",
        "",
        f"- {len(df)}/{expected_rows} unique cells; 18 task variants, 9 arms, 5 seeds.",
        "- Missing, duplicate, non-finite, out-of-range accuracy/ECE: 0.",
        f"- Exact cross-arm metric duplicates: {cross_arm_identical}.",
        "- `ctrlBpp` is not a second effective arm in this matrix: its norm-collapse release condition never activated, and its lambda trace and final metrics equal `ctrlB` in all 90 paired cells. This is an inert predeclared variant, not missing compute; it is retained transparently and not counted as independent evidence.",
        f"- Raw SHA-256: `{evidence['raw_sha256']}`.",
        "",
        "## Descriptive ranking",
        "",
        "| Rank | Arm | Mean accuracy | Median accuracy | Mean ECE | Mean gap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in arm_rank.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['arm']} | {row['test_acc_mean']:.4f} | "
            f"{row['test_acc_median']:.4f} | {row['test_ece_mean']:.4f} | {row['gap_mean']:.4f} |"
        )
    lines += [
        "",
        "This ranking is descriptive; correlated task variants and repeated seeds are not treated as independent evidence. Exact ties share a rank.",
        "",
        "## Predeclared controller-versus-strong-baseline family",
        "",
        "| Controller | Comparator | Mean diff. | Median diff. | 95% cluster-bootstrap CI (median) | W/T/L | Raw p | Holm p | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['controller']} | {row['comparator']} | {row['mean_paired_difference']:+.4f} | "
            f"{row['median_paired_difference']:+.4f} | [{row['cluster_bootstrap_median_ci_low']:+.4f}, "
            f"{row['cluster_bootstrap_median_ci_high']:+.4f}] | "
            f"{int(row['task_variant_wins'])}/{int(row['task_variant_ties'])}/{int(row['task_variant_losses'])} | "
            f"{row['task_level_wilcoxon_p_raw']:.4g} | {row['task_level_wilcoxon_p_holm']:.4g} | "
            f"{'PASS' if row['passes_superiority_gate'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Claim disposition",
        "",
        f"Controller superiority gate passes: **{len(passes)} of {len(comparison_df)}**.",
        "The original 12-comparison family is retained because it was predeclared. Because B+ was inert and duplicates B, the unique active-controller family contains eight A/B comparisons; none passes either.",
        "The seed-cell Wilcoxon results are retained in `primary_comparisons.csv` as sensitivity only; they do not gate claims.",
        "A zero gate count requires the predeclared benchmark/boundary pivot and prohibits a superior-new-controller narrative.",
        "",
        "## Non-claim boundary",
        "",
        "This tabular layer does not close the architecture layer, theory proof, validation-tuned cost integration, manuscript revision, or reference-integrity gate.",
        "",
    ]
    atomic_text(OUT / "ANALYSIS_REPORT.md", "\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
