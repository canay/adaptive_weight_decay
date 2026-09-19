#!/usr/bin/env python3
"""Rebuild all statistics for SCI-f06 from the clean audit rerun.

Reads results/audit_rerun_<stamp>/raw_runs.csv and recomputes every quantity the
manuscript reports, plus the audit-required additions:
  * per-task oracle / median / controller(lambda0=0) means for acc, gap, ECE
  * Wilcoxon signed-rank tests (default vs oracle/median, same-lambda0, pooled)
    with rank-biserial correlation and Cohen's d_z
  * Holm-Bonferroni adjusted p for the two confirmatory pooled tests
  * pooled ECE and pooled gap contrasts (so accuracy gains and calibration loss
    appear on equal footing)
  * sensitivity IQR and best-minus-median over the 7-point grid and the lambda<=10 subgrid
  * baseline contrasts (AdaDecay/AdamP/cosine/step) vs the controller
  * learning-rate transfer summary

Writes results/audit_rerun_<stamp>/stats_audit.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
SUBGRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0]  # excludes degenerate 30
TASKS = ["adult_full", "bank_full", "digits_full", "adult_n20", "bank_n20", "digits_n20"]
SEEDS = [0, 1, 2, 3, 4]
BASELINES = ["adadecay_default", "adadecay_regime", "adamp_proj", "cosine_wd", "step_early_wd"]


def paired_stats(ctrl: np.ndarray, ref: np.ndarray) -> dict:
    """Wilcoxon signed-rank with rank-biserial and Cohen d_z. Positive favors ctrl."""
    d = np.asarray(ctrl, float) - np.asarray(ref, float)
    nz = d[d != 0]
    n = len(d)
    if len(nz) == 0:
        return dict(p=1.0, W=0.0, rrb=0.0, dz=0.0, n=n)
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    wp = float(ranks[nz > 0].sum())
    wm = float(ranks[nz < 0].sum())
    rrb = (wp - wm) / (wp + wm)
    try:
        p = float(wilcoxon(ctrl, ref).pvalue)
    except ValueError:
        p = 1.0
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    return dict(p=p, W=float(min(wp, wm)), rrb=float(rrb), dz=dz, n=int(n))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        running = max(running, a)
        adj[k] = running
    return adj


def seed_means(df, task, method, lam0, lr, col):
    """Per-seed value vector for a given cell (already one row per seed)."""
    sub = df[(df.task == task) & (df.method == method) & (df.lr == lr) &
             (np.isclose(df.lam0.fillna(-999), lam0 if lam0 is not None else -999))]
    sub = sub[~sub.get("diverged", False).astype(bool)] if "diverged" in sub else sub
    return sub.sort_values("seed")[col].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    df = pd.read_csv(d / "raw_runs.csv")
    if "diverged" not in df:
        df["diverged"] = False
    df["diverged"] = df["diverged"].fillna(False).astype(bool)
    lr0 = 3e-3
    seeds_present = sorted(int(s) for s in df["seed"].unique())

    out = {"grid": GRID, "n_seeds": len(SEEDS), "tasks": {}}

    # per-task references and tests
    pooled = {k: {"ctrl": [], "oracle": [], "median": []} for k in ["acc", "gap", "ece"]}
    pooled6_acc = {"ctrl": [], "oracle": []}
    col = {"acc": "test_acc", "gap": "gap", "ece": "test_ece"}
    for task in TASKS:
        fixed = df[(df.task == task) & (df.method == "fixed") & (df.lr == lr0) & (~df.diverged)]
        grid_means = {lam: fixed[np.isclose(fixed.lam0, lam)]["test_acc"].mean() for lam in GRID}
        lam_star = max(grid_means, key=grid_means.get)
        med_val = np.median([grid_means[l] for l in GRID])
        lam_median = min(GRID, key=lambda l: abs(grid_means[l] - med_val))

        rec = {"lam_star": lam_star, "lam_median": lam_median, "grid_acc_means": grid_means,
               "tests": {}}
        for metric, c in col.items():
            o = seed_means(df, task, "fixed", lam_star, lr0, c)
            m = seed_means(df, task, "fixed", lam_median, lr0, c)
            ct = seed_means(df, task, "ctrl", 0.0, lr0, c)
            rec[f"oracle_{metric}_mean"] = float(np.mean(o))
            rec[f"median_{metric}_mean"] = float(np.mean(m))
            rec[f"ctrl_{metric}_mean"] = float(np.mean(ct))
            pooled[metric]["ctrl"].extend(ct.tolist())
            pooled[metric]["oracle"].extend(o.tolist())
            pooled[metric]["median"].extend(m.tolist())
            if metric == "acc":
                rec["tests"]["default_vs_oracle"] = paired_stats(ct, o)
                rec["tests"]["default_vs_median"] = paired_stats(ct, m)

        # controller pooled over 6 nondegenerate starts, paired by seed vs oracle
        c6 = []
        for s in seeds_present:
            vals = [df[(df.task == task) & (df.method == "ctrl") & (df.lr == lr0) &
                       (df.seed == s) & (np.isclose(df.lam0, l)) & (~df.diverged)]["test_acc"].mean()
                    for l in SUBGRID]
            c6.append(np.nanmean(vals))
        rec["ctrl_pooled6_acc_mean"] = float(np.mean(c6))
        o_acc = seed_means(df, task, "fixed", lam_star, lr0, "test_acc")
        pooled6_acc["ctrl"].extend(np.array(c6).tolist())
        pooled6_acc["oracle"].extend(o_acc.tolist())

        # same-lambda0 paired (ctrl vs fixed at equal lambda), n=35
        cc, ff = [], []
        for lam in GRID:
            for s in seeds_present:
                cv = df[(df.task == task) & (df.method == "ctrl") & (df.lr == lr0) &
                        (df.seed == s) & (np.isclose(df.lam0, lam)) & (~df.diverged)]["test_acc"]
                fv = df[(df.task == task) & (df.method == "fixed") & (df.lr == lr0) &
                        (df.seed == s) & (np.isclose(df.lam0, lam)) & (~df.diverged)]["test_acc"]
                if len(cv) and len(fv):
                    cc.append(float(cv.mean()))
                    ff.append(float(fv.mean()))
        rec["tests"]["ctrl_samelam0_vs_fixed_paired"] = paired_stats(np.array(cc), np.array(ff))

        # sensitivity: IQR and best-minus-median over grids (seed-mean accuracies)
        def spread(method, lams):
            sm = np.array([df[(df.task == task) & (df.method == method) & (df.lr == lr0) &
                              (np.isclose(df.lam0, l)) & (~df.diverged)]["test_acc"].mean() for l in lams])
            q75, q25 = np.percentile(sm, [75, 25])
            return dict(iqr=float(q75 - q25), best_minus_median=float(sm.max() - np.median(sm)))
        rec["sensitivity"] = {
            "fixed_7pt": spread("fixed", GRID), "ctrl_7pt": spread("ctrl", GRID),
            "fixed_sub": spread("fixed", SUBGRID), "ctrl_sub": spread("ctrl", SUBGRID),
        }
        out["tasks"][task] = rec

    # pooled tests (n=30) for acc/gap/ece
    out["pooled"] = {}
    out["pooled_means"] = {}
    for metric in ["acc", "gap", "ece"]:
        ct = np.array(pooled[metric]["ctrl"])
        orc = np.array(pooled[metric]["oracle"])
        med = np.array(pooled[metric]["median"])
        out["pooled"][f"{metric}_default_vs_oracle"] = paired_stats(ct, orc)
        out["pooled"][f"{metric}_default_vs_median"] = paired_stats(ct, med)
        out["pooled_means"][metric] = dict(ctrl=float(ct.mean()), oracle=float(orc.mean()),
                                           median=float(med.mean()))
    out["pooled"]["acc_pooled6_vs_oracle"] = paired_stats(np.array(pooled6_acc["ctrl"]),
                                                          np.array(pooled6_acc["oracle"]))

    # Holm on the two confirmatory pooled accuracy tests
    conf = {"acc_default_vs_oracle": out["pooled"]["acc_default_vs_oracle"]["p"],
            "acc_default_vs_median": out["pooled"]["acc_default_vs_median"]["p"]}
    out["holm_pooled_confirmatory"] = holm(conf)

    # baselines: per-task mean metrics + paired vs controller(default) on accuracy
    out["baselines"] = {}
    for task in TASKS:
        ct = seed_means(df, task, "ctrl", 0.0, lr0, "test_acc")
        brec = {}
        for b in BASELINES:
            sub = df[(df.task == task) & (df.method == b) & (df.lr == lr0) & (~df.diverged)].sort_values("seed")
            if not len(sub):
                continue
            ba = sub["test_acc"].to_numpy()
            brec[b] = dict(acc_mean=float(ba.mean()), gap_mean=float(sub["gap"].mean()),
                           ece_mean=float(sub["test_ece"].mean()),
                           lam_final_mean=float(sub["lam_final"].mean()),
                           vs_ctrl=paired_stats(ct, ba) if len(ba) == len(ct) else None)
        out["baselines"][task] = brec

    # transfer summary (noisy tasks, lr in {1e-3,1e-2})
    out["transfer"] = {}
    for task in ["adult_n20", "bank_n20", "digits_n20"]:
        for lr in [1e-3, 1e-2]:
            fx = df[(df.task == task) & (df.method == "fixed") & (df.lr == lr) & (~df.diverged)]
            if not len(fx):
                continue
            gm = {lam: fx[np.isclose(fx.lam0, lam)]["test_acc"].mean() for lam in GRID}
            lam_star = max(gm, key=gm.get)
            med_val = np.median(list(gm.values()))
            lam_med = min(GRID, key=lambda l: abs(gm[l] - med_val))
            ct = seed_means(df, task, "ctrl", 0.0, lr, "test_acc")
            out["transfer"][f"{task}@{lr}"] = dict(
                oracle=float(gm[lam_star]), median=float(gm[lam_med]),
                ctrl_default=float(ct.mean()) if len(ct) else None, lam_star=lam_star)

    Path(d / "stats_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # console summary
    pm = out["pooled_means"]["acc"]
    print(f"pooled acc: ctrl={pm['ctrl']:.4f} oracle={pm['oracle']:.4f} median={pm['median']:.4f}")
    print(f"pooled vs oracle p={out['pooled']['acc_default_vs_oracle']['p']:.2e} "
          f"vs median p={out['pooled']['acc_default_vs_median']['p']:.4f}")
    print(f"holm: {out['holm_pooled_confirmatory']}")
    print(f"-> {d/'stats_audit.json'}")


if __name__ == "__main__":
    main()
