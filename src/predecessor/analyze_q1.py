#!/usr/bin/env python3
"""Consolidated analysis of the Q1 suite: per-dataset means, decisive pooled
significance (paired across 18 datasets x 5 seeds), regime split, and the
budget/efficiency comparison (controller vs validation tuning)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
TAB = ["adult", "bank", "digits", "phoneme", "spambase", "qsar", "waveform", "mfeat", "satimage"]


def paired(df, a, b):
    """Paired Wilcoxon of method a vs b across (dataset, seed) cells."""
    xa, xb = [], []
    for name in df.name.unique():
        for sd in sorted(df.seed.unique()):
            va = df[(df.name == name) & (df.method == a) & (df.seed == sd)]["test_acc"]
            vb = df[(df.name == name) & (df.method == b) & (df.seed == sd)]["test_acc"]
            if len(va) and len(vb):
                xa.append(float(va.mean())); xb.append(float(vb.mean()))
    xa, xb = np.array(xa), np.array(xb)
    d = xa - xb
    try:
        p = float(wilcoxon(xa, xb).pvalue)
    except ValueError:
        p = 1.0
    return dict(mean_a=float(xa.mean()), mean_b=float(xb.mean()), mean_delta=float(d.mean()),
                win_rate=float((d > 0).mean()), p=p, n=len(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    d = Path(a.dir)
    df = pd.read_csv(d / "q1_raw.csv")
    df = df[~df["diverged"].fillna(False)]
    names = [f"{t}_{v}" for v in ("full", "n20") for t in TAB if f"{t}_{v}" in set(df.name)]
    # Public-export repair: keep the frozen tabular comparison scope (18 x 5).
    # The mixed breadth archive also contains CNN rows; they are a separate layer.
    df = df[df.name.isin(names)].copy()

    def cell(name, method, lam=None):
        s = df[(df.name == name) & (df.method == method)]
        if lam is not None:
            s = s[np.isclose(s.lam0, lam)]
        return float(s["test_acc"].mean())

    per = {}
    for name in names:
        fixed = {l: cell(name, "fixed", l) for l in GRID}
        per[name] = dict(oracle=max(fixed.values()), median=float(np.median(list(fixed.values()))),
                         ctrlA=cell(name, "ctrlA"), ctrlB=cell(name, "ctrlB"),
                         adadecay=cell(name, "adadecay_regime"), valtuned=cell(name, "valtuned"))
    pm = lambda k: float(np.mean([per[n][k] for n in names]))
    pooled_means = {k: pm(k) for k in ["oracle", "median", "ctrlA", "ctrlB", "adadecay", "valtuned"]}

    tests = {
        "ctrlB_vs_ctrlA": paired(df, "ctrlB", "ctrlA"),
        "ctrlB_vs_adadecay": paired(df, "ctrlB", "adadecay_regime"),
        "ctrlB_vs_valtuned": paired(df, "ctrlB", "valtuned"),
    }
    # regime split
    clean = [n for n in names if n.endswith("_full")]
    noisy = [n for n in names if n.endswith("_n20")]
    regime = {}
    for grp, ns in [("clean", clean), ("noisy", noisy)]:
        regime[grp] = {k: float(np.mean([per[n][k] for n in ns]))
                       for k in ["oracle", "median", "ctrlA", "ctrlB", "adadecay", "valtuned"]}
        regime[grp]["ctrlB_beats_ctrlA"] = int(sum(per[n]["ctrlB"] > per[n]["ctrlA"] for n in ns))
        regime[grp]["n"] = len(ns)

    out = dict(per_dataset=per, pooled_means=pooled_means, tests=tests, regime=regime,
               budget=dict(ctrlB_runs=1, valtuned_runs=len(GRID) + 1,
                           ctrlB_acc=pooled_means["ctrlB"], valtuned_acc=pooled_means["valtuned"],
                           cost_ratio=len(GRID) + 1))
    (d / "stats_q1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("POOLED (", len(names), "datasets):")
    for k, v in pooled_means.items():
        print(f"  {k:10s} {v:.4f}")
    print("PAIRED (n=90):")
    for k, v in tests.items():
        print(f"  {k:20s} delta={v['mean_delta']:+.4f} win={v['win_rate']:.0%} p={v['p']:.2e} n={v['n']}")
    print("REGIME clean: ctrlA %.3f ctrlB %.3f (B>A on %d/%d)" %
          (regime["clean"]["ctrlA"], regime["clean"]["ctrlB"], regime["clean"]["ctrlB_beats_ctrlA"], regime["clean"]["n"]))
    print("REGIME noisy: ctrlA %.3f ctrlB %.3f (B>A on %d/%d)" %
          (regime["noisy"]["ctrlA"], regime["noisy"]["ctrlB"], regime["noisy"]["ctrlB_beats_ctrlA"], regime["noisy"]["n"]))
    print("-> stats_q1.json")


if __name__ == "__main__":
    main()
