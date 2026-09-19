#!/usr/bin/env python3
"""Budget-matched comparison: validation-based decay tuning vs the controller.

The oracle in the main study is a non-deployable upper bound. The realistic
comparator a practitioner faces is validation-based tuning: hold out a validation
split, sweep the decay grid, pick the best lambda, and (optionally) retrain on the
full training set. That procedure costs |grid| training runs plus a validation
split. The controller costs one run and no validation labels.

This script measures test accuracy AND cost (training runs, validation labels) for:
  val_tuned   : 80/20 split, sweep grid on val, retrain on full train with best lambda
  controller  : full train, lambda0=0, no validation
  median_fixed: the untuned grid-median lambda, one run, no validation
so the methods can be compared on the accuracy-vs-cost frontier.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common6 as C  # noqa: E402

torch.set_num_threads(1)
C.TASKS["adult_n20"] = dict(src="adult", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["bank_n20"] = dict(src="bank", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["digits_n20"] = dict(src="digits", n_sub=500, epochs=40, batch=128, noise=0.2)

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
LR = 3e-3


def train_fixed(Xtr, ytr, Xev, yev, n_class, lam, epochs, batch, seed):
    model = C.make_mlp(Xtr.shape[1], n_class, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=lam)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(60000 + seed)
    nrm = len(Xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(nrm, generator=g)
        for i in range(0, nrm, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
    return C.evaluate(model, Xev, yev)["acc"]


def val_tuned(task, seed):
    cfg = C.TASKS[task]
    D = C.load_task(task)
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    n_class = D["n_class"]
    ep, ba = cfg["epochs"], cfg.get("batch", C.BATCH)
    rng = np.random.RandomState(777 + seed)
    n = len(Xtr)
    idx = rng.permutation(n)
    nval = int(0.2 * n)
    vi, ti = idx[:nval], idx[nval:]
    best_lam, best_acc = None, -1.0
    for lam in GRID:
        a = train_fixed(Xtr[ti], ytr[ti], Xtr[vi], ytr[vi], n_class, lam, ep, ba, seed)
        if a > best_acc:
            best_acc, best_lam = a, lam
    # retrain on full train with the selected lambda
    test_acc = train_fixed(Xtr, ytr, Xte, yte, n_class, best_lam, ep, ba, seed)
    return test_acc, best_lam, len(GRID) + 1  # n_runs


def controller(task, seed):
    r = C.run_unit(dict(task=task, method="ctrl", lam0=0.0, lr=LR, seed=seed,
                        rule="A", eta=1.0, beta=0.5, warmup=2))
    return r["test_acc"], 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="adult_full,bank_full,digits_full,adult_n20,bank_n20,digits_n20")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")
    print(f"{'task':12s} {'val_tuned':>10s} {'ctrl':>8s} {'ctrl-vt':>8s} {'vt_runs':>8s} {'ctrl_runs':>9s}")
    import json
    out = {}
    for t in tasks:
        vt = np.array([val_tuned(t, s) for s in seeds], dtype=object)
        vt_acc = np.array([x[0] for x in vt]); vt_runs = vt[0][2]
        c = np.array([controller(t, s) for s in seeds], dtype=object)
        c_acc = np.array([x[0] for x in c])
        delta = c_acc.mean() - vt_acc.mean()
        out[t] = dict(val_tuned_acc=float(vt_acc.mean()), ctrl_acc=float(c_acc.mean()),
                      delta=float(delta), vt_runs=int(vt_runs), ctrl_runs=1,
                      vt_best_lams=[x[1] for x in vt])
        print(f"{t:12s} {vt_acc.mean():10.4f} {c_acc.mean():8.4f} {delta:+8.4f} {vt_runs:8d} {1:9d}")
    Path("results/budget_hypothesis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/budget_hypothesis.json")


if __name__ == "__main__":
    main()
