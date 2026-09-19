#!/usr/bin/env python3
"""Audit reruns + close-prior baselines for SCI-f06 (resumable, parallel, one command).

Reuses the authoritative main-stage harness in ``common6.py`` so the fixed-decay grid
and the norm-growth controller are reproduced from the same data loaders, model
initialization, seeds, and evaluation. Adds the close-prior baselines requested in the
Q1 audit:

  * AdaDecay  (Nakamura & Hong, 2019) ported to AdamW. Per-parameter decoupled decay
    lambda*theta_j with theta_j = 2/(1+exp(-alpha*zbar_j)) and zbar_j the per-tensor
    standardized absolute data gradient (Eqs. 9, 10, 12). Defaults alpha=4, lambda=5e-4;
    a regime-matched lambda=1.0 variant is also run for fairness in this decay range.
  * AdamP     (Heo et al., 2021) via the official ``adamp`` package, weight_decay=0 so
    the projection alone controls norm growth (the analog of the controller from lambda0=0).
  * cosine    decoupled weight-decay schedule, lambda_max -> 0 over training.
  * step      early weight decay (lambda_max for the first half, then 0).

Schedule magnitude is a single cross-task validation-free default (lambda_max=1.0).

Execution model: every (task, method, lr, seed, lambda0) cell is computed once and cached
as results/<out>/runs/<key>.json (atomic write). Re-running skips completed cells, so the
suite can be run foreground in short chunks until it finishes. A ProcessPoolExecutor runs
cells in parallel (CPU, one thread per worker). When all cells exist, raw_runs.csv,
controller_traces.csv, and manifest.json are aggregated.

Usage:
  python code/run_experiments_audit.py                 # full suite (resumable)
  python code/run_experiments_audit.py --workers 6
  python code/run_experiments_audit.py --quick         # 1 seed sanity check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common6 as C  # noqa: E402

torch.set_num_threads(1)

# canonical main noisy tasks (close the provenance gap: make them first-class)
C.TASKS["adult_n20"] = dict(src="adult", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["bank_n20"] = dict(src="bank", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["digits_n20"] = dict(src="digits", n_sub=500, epochs=40, batch=128, noise=0.2)

MAIN_TASKS = ["adult_full", "bank_full", "digits_full", "adult_n20", "bank_n20", "digits_n20"]
NOISY_TASKS = ["adult_n20", "bank_n20", "digits_n20"]
GRID = list(C.GRID)
CENTRAL_LR = 3e-3
TRANSFER_LRS = [1e-3, 1e-2]
TRANSFER_CTRL_STARTS = [0.0, 0.1, 10.0]

BASELINES = {
    "adadecay_default": dict(method="adadecay", alpha=4.0, lam_base=5e-4),
    "adadecay_regime": dict(method="adadecay", alpha=4.0, lam_base=1.0),
    "adamp_proj": dict(method="adamp", wd=0.0),
    "cosine_wd": dict(method="cosine", lam_max=1.0),
    "step_early_wd": dict(method="step", lam_max=1.0),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def train_baseline(task: str, seed: int, method: str, lr: float = CENTRAL_LR, **hp) -> dict:
    """Baseline loop mirroring common6.run_unit (same init seed, batch generator, eval)."""
    cfg = C.TASKS[task]
    D = C.load_task(task)
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    n = len(Xtr)
    model = C.make_mlp(Xtr.shape[1], D["n_class"], seed)
    batch = cfg.get("batch", C.BATCH)
    g = torch.Generator().manual_seed(60000 + seed)
    lossf = nn.CrossEntropyLoss()

    alpha = float(hp.get("alpha", 4.0))
    lam_base = float(hp.get("lam_base", 5e-4))
    lam_max = float(hp.get("lam_max", 1.0))
    T = cfg["epochs"]

    if method == "adamp":
        from adamp import AdamP
        wd = float(hp.get("wd", 0.0))
        opt = AdamP(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=wd, delta=0.1, wd_ratio=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.0)

    hist = []
    wn_prev = C.param_norm(model)
    diverged = False
    for ep in range(T):
        model.train()
        if method == "cosine":
            lam_ep = lam_max * 0.5 * (1.0 + math.cos(math.pi * ep / max(1, T - 1)))
            opt.param_groups[0]["weight_decay"] = lam_ep
        elif method == "step":
            lam_ep = lam_max if ep < T // 2 else 0.0
            opt.param_groups[0]["weight_decay"] = lam_ep
        elif method == "adadecay":
            lam_ep = lam_base
        else:  # adamp
            lam_ep = float(opt.param_groups[0]["weight_decay"])

        perm = torch.randperm(n, generator=g)
        losses = []
        n_steps = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), ytr[idx])
            lv = float(loss.detach())
            if not math.isfinite(lv):
                diverged = True
                break
            loss.backward()
            if method == "adadecay":
                thetas = []
                with torch.no_grad():
                    for p in model.parameters():
                        gj = p.grad.detach().abs()
                        mu = gj.mean()
                        sd = gj.std(unbiased=False) + 1e-12
                        thetas.append(2.0 / (1.0 + torch.exp(-alpha * (gj - mu) / sd)))
                opt.step()
                with torch.no_grad():
                    for p, theta in zip(model.parameters(), thetas):
                        p.mul_(1.0 - lr * lam_base * theta)
            else:
                opt.step()
            losses.append(lv)
            n_steps += 1
        if diverged:
            break
        wn = C.param_norm(model)
        gr = (wn - wn_prev) / (wn_prev + 1e-12)
        wn_prev = wn
        ei = D["tr_eval_idx"]
        etr = C.evaluate(model, Xtr[ei], ytr[ei])
        ete = C.evaluate(model, Xte, yte)
        hist.append(dict(ep=ep, lam=float(lam_ep), train_acc=etr["acc"],
                         test_acc=ete["acc"], gap=etr["acc"] - ete["acc"], wnorm=wn, g=gr))
    if diverged or not hist:
        return dict(diverged=True, hist=hist)
    ei = D["tr_eval_idx"]
    etr = C.evaluate(model, Xtr[ei], ytr[ei])
    ete = C.evaluate(model, Xte, yte)
    return dict(diverged=False, test_acc=ete["acc"], test_loss=ete["loss"],
                test_ece=ete["ece"], train_acc=etr["acc"], train_loss=etr["loss"],
                gap=etr["acc"] - ete["acc"], lam_final=hist[-1]["lam"], hist=hist)


def cell_key(task, method, lr, seed, lam0) -> str:
    s = f"{task}__{method}__lr{lr}__s{seed}__l{lam0}"
    return s.replace(".", "p").replace("-", "m")


def run_one_cell(args) -> str:
    """Worker: compute a single cell and atomically cache it. Returns the key."""
    task, method, lr, seed, lam0, runs_dir = args
    runs_dir = Path(runs_dir)
    key = cell_key(task, method, lr, seed, lam0)
    out = runs_dir / f"{key}.json"
    if out.exists():
        return key
    torch.set_num_threads(1)
    if method == "fixed":
        r = C.run_unit(dict(task=task, method="fixed", lam0=lam0, lr=lr, seed=seed))
    elif method == "ctrl":
        r = C.run_unit(dict(task=task, method="ctrl", lam0=lam0, lr=lr, seed=seed,
                            rule="A", eta=1.0, beta=0.5, warmup=2))
    else:
        r = train_baseline(task, seed, lr=lr, **BASELINES[method])
    row = dict(task=task, method=method, lr=lr, seed=seed, lam0=lam0,
               diverged=bool(r.get("diverged", False)))
    if not row["diverged"]:
        for k in ("test_acc", "test_loss", "test_ece", "train_acc", "gap", "lam_final"):
            if k in r:
                row[k] = r[k]
    payload = dict(row=row, hist=r.get("hist", []))
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return key


def plan_cells(seeds, quick):
    cells = []
    for task in MAIN_TASKS:
        for seed in seeds:
            for lam in GRID:
                cells.append((task, "fixed", CENTRAL_LR, seed, lam))
            for lam0 in GRID:
                cells.append((task, "ctrl", CENTRAL_LR, seed, lam0))
    for task in MAIN_TASKS:
        for seed in seeds:
            for bname in BASELINES:
                cells.append((task, bname, CENTRAL_LR, seed, None))
    if not quick:
        for task in NOISY_TASKS:
            for lr in TRANSFER_LRS:
                for seed in seeds:
                    for lam in GRID:
                        cells.append((task, "fixed", lr, seed, lam))
                    for lam0 in TRANSFER_CTRL_STARTS:
                        cells.append((task, "ctrl", lr, seed, lam0))
    return cells


def aggregate(out_dir: Path, runs_dir: Path, seeds, t0):
    rows, traces = [], []
    for f in sorted(runs_dir.glob("*.json")):
        payload = json.loads(f.read_text())
        rows.append(payload["row"])
        for h in payload["hist"]:
            traces.append(dict(task=payload["row"]["task"], method=payload["row"]["method"],
                               lr=payload["row"]["lr"], seed=payload["row"]["seed"],
                               lam0=payload["row"]["lam0"], **h))
    raw = pd.DataFrame(rows)
    trace_df = pd.DataFrame(traces)
    raw.to_csv(out_dir / "raw_runs.csv", index=False)
    trace_df.to_csv(out_dir / "controller_traces.csv", index=False)
    code_dir = Path(__file__).resolve().parent
    manifest = dict(
        generated_utc=utc_now(), wall_seconds=round(time.time() - t0, 1),
        python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
        scipy=__import__("scipy").__version__, sklearn=__import__("sklearn").__version__,
        seeds=seeds, central_lr=CENTRAL_LR, transfer_lrs=TRANSFER_LRS, grid=GRID,
        main_tasks=MAIN_TASKS, baselines=BASELINES,
        controller=dict(rule="A", eta=1.0, beta=0.5, warmup=2, clip=[0.0, 30.0]),
        n_raw_rows=int(len(raw)), n_diverged=int(raw["diverged"].sum()) if "diverged" in raw else 0,
        n_trace_rows=int(len(trace_df)),
        code_sha256={p.name: sha256(p) for p in [code_dir / "common6.py",
                     code_dir / "run_experiments_audit.py"]},
        output_sha256={"raw_runs.csv": sha256(out_dir / "raw_runs.csv"),
                       "controller_traces.csv": sha256(out_dir / "controller_traces.csv")},
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stamp", default="20260623")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.quick:
        seeds = [0]
    else:
        seeds = [0, 1, 2, 3, 4]

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else root / "results" / f"audit_rerun_{args.stamp}"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cells = plan_cells(seeds, args.quick)
    todo = [(*c, str(runs_dir)) for c in cells if not (runs_dir / f"{cell_key(*c)}.json").exists()]
    done0 = len(cells) - len(todo)
    print(f"plan: {len(cells)} cells, {done0} cached, {len(todo)} to run, {args.workers} workers", flush=True)

    completed = 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one_cell, a) for a in todo]
            for fu in as_completed(futs):
                fu.result()
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    print(f"  {completed}/{len(todo)} cells done ({time.time()-t0:.0f}s)", flush=True)

    n_cached = sum(1 for c in cells if (runs_dir / f"{cell_key(*c)}.json").exists())
    if n_cached == len(cells):
        man = aggregate(out_dir, runs_dir, seeds, t0)
        print(f"DONE  rows={man['n_raw_rows']} diverged={man['n_diverged']} "
              f"traces={man['n_trace_rows']} -> {out_dir} ({man['wall_seconds']}s)", flush=True)
    else:
        print(f"PARTIAL {n_cached}/{len(cells)} cached; re-run to resume.", flush=True)


if __name__ == "__main__":
    main()
