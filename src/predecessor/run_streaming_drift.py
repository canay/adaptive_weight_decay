#!/usr/bin/env python3
"""Delayed-label prequential (test-then-train) evaluation under concept drift.

This exercises the validation-free controller in the deployment setting the paper
motivates: data arrive as an ordered stream, labels are revealed only after a
prediction is required, and the concept drifts, so a validation-based decay sweep
is not available. The same MLP, AdamW, and norm-growth controller (rule A) are
used; each stream window plays the role of an epoch for the controller signal.

Streams:
  electricity          real elec2 benchmark (OpenML 'electricity'), price up/down
  rotating_hyperplane  synthetic gradual drift (Hulten et al., 2001)
  sea                  synthetic sudden drift (Street & Kim, 2001)

Protocol per window of W ordered instances:
  1. PREDICT the window with the current model (prequential test; labels delayed).
  2. TRAIN on the window for a few inner passes once labels are revealed.
  3. Update the controller from the window's weight-norm growth and adjust lambda.

Arms: fixed decay grid, controller (lambda0=0), AdaDecay (default + regime).
Post-hoc references: oracle = best-prequential fixed lambda (not deployable),
median = median fixed lambda. Output is resumable (per-cell JSON) and parallel.

Usage:
  python code/run_streaming_drift.py            # full (5 seeds)
  python code/run_streaming_drift.py --quick
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

STREAMS = ["electricity", "rotating_hyperplane", "sea"]
GRID = [0.0, 0.01, 0.1, 1.0, 10.0]
LR = 3e-3
WINDOW = 1000
INNER_EPOCHS = 3
BATCH = 128
WARMUP = 2

BASELINES = {
    "adadecay_default": dict(method="adadecay", alpha=4.0, lam_base=5e-4),
    "adadecay_regime": dict(method="adadecay", alpha=4.0, lam_base=1.0),
}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- streams
def make_stream(name, seed):
    """Return (X float32 standardized by first-window stats, y int64, n_class, n)."""
    if name == "electricity":
        from sklearn.datasets import fetch_openml
        d = fetch_openml("electricity", version=1, as_frame=True)
        df = d.data.copy()
        # numeric features in temporal order; drop the raw date index
        cols = [c for c in df.columns if c != "date"]
        X = df[cols].astype(float).to_numpy()
        y = (d.target.to_numpy() == "UP").astype(np.int64)
        # electricity is a fixed real stream; the seed only varies model init
    elif name == "rotating_hyperplane":
        rng = np.random.RandomState(1000 + seed)
        n, dim = 20000, 10
        X = rng.rand(n, dim)
        a = rng.rand(dim) + 0.5            # positive weights
        drift_dims = dim // 2
        sigma = 0.001                      # slow continuous rotation
        direction = np.ones(drift_dims)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            a[:drift_dims] += sigma * direction
            # reflect at bounds to keep a rotating, bounded drift
            for j in range(drift_dims):
                if a[j] > 1.5 or a[j] < 0.2:
                    direction[j] *= -1
            thr = 0.5 * a.sum()
            label = 1 if X[i] @ a >= thr else 0
            if rng.rand() < 0.05:          # 5% label noise
                label = 1 - label
            y[i] = label
    elif name == "sea":
        rng = np.random.RandomState(2000 + seed)
        n = 20000
        X = rng.rand(n, 3) * 10.0
        thresholds = [8.0, 9.0, 7.0, 9.5]  # sudden drift at block boundaries
        block = n // len(thresholds)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            theta = thresholds[min(i // block, len(thresholds) - 1)]
            label = 1 if (X[i, 0] + X[i, 1]) <= theta else 0
            if rng.rand() < 0.10:          # 10% class noise (SEA convention)
                label = 1 - label
            y[i] = label
    else:
        raise ValueError(name)
    # standardize using the first window only (no future leakage)
    w = min(WINDOW, len(X))
    mu, sd = X[:w].mean(0), X[:w].std(0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)
    return torch.tensor(Xs), torch.tensor(y), int(y.max() + 1), len(y)


# ------------------------------------------------------------------------- one arm
def run_stream_arm(name, init_seed, method, lam0=0.0, **hp):
    X, y, n_class, n = make_stream(name, init_seed)
    model = C.make_mlp(X.shape[1], n_class, init_seed)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(70000 + init_seed)
    alpha = float(hp.get("alpha", 4.0))
    lam_base = float(hp.get("lam_base", 5e-4))
    is_adadecay = method == "adadecay"
    lam = 0.0 if is_adadecay else float(lam0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=lam)
    wn_prev = C.param_norm(model)
    gbar = 0.0
    correct = total = 0
    per_window = []
    widx = 0
    for start in range(0, n, WINDOW):
        Xw, yw = X[start:start + WINDOW], y[start:start + WINDOW]
        if len(Xw) < 2:
            break
        # 1) prequential prediction (labels delayed)
        model.eval()
        with torch.no_grad():
            logits = model(Xw)
            pred = logits.argmax(1)
            cw = int((pred == yw).sum())
        correct += cw
        total += len(yw)
        acc_w = cw / len(yw)
        # 2) train on the window once labels are revealed
        model.train()
        nsteps = 0
        for _ in range(INNER_EPOCHS):
            perm = torch.randperm(len(Xw), generator=g)
            for i in range(0, len(Xw), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(Xw[idx]), yw[idx])
                if not torch.isfinite(loss):
                    break
                loss.backward()
                if is_adadecay:
                    thetas = []
                    with torch.no_grad():
                        for p in model.parameters():
                            gj = p.grad.detach().abs()
                            z = (gj - gj.mean()) / (gj.std(unbiased=False) + 1e-12)
                            thetas.append(2.0 / (1.0 + torch.exp(-alpha * z)))
                    opt.step()
                    with torch.no_grad():
                        for p, th in zip(model.parameters(), thetas):
                            p.mul_(1.0 - LR * lam_base * th)
                else:
                    opt.step()
                nsteps += 1
        # 3) controller signal update from window norm growth
        wn = C.param_norm(model)
        gr = (wn - wn_prev) / (wn_prev + 1e-12)
        wn_prev = wn
        gbar = 0.5 * gbar + 0.5 * gr
        s = gbar / (LR * max(1, nsteps))
        if method == "ctrl" and widx + 1 >= WARMUP:
            lam = min(max(lam + 1.0 * min(max(s, -1.0), 1.0), 0.0), 30.0)
            opt.param_groups[0]["weight_decay"] = lam
        per_window.append(dict(window=widx, acc=acc_w, lam=float(lam if not is_adadecay else lam_base),
                               wnorm=wn, s=float(s)))
        widx += 1
    return dict(prequential_acc=correct / max(1, total), n_windows=widx, per_window=per_window)


def cell_key(stream, method, lam0, seed):
    return f"{stream}__{method}__l{lam0}__s{seed}".replace(".", "p").replace("-", "m")


def run_one_cell(args):
    stream, method, lam0, seed, runs_dir = args
    runs_dir = Path(runs_dir)
    out = runs_dir / f"{cell_key(stream, method, lam0, seed)}.json"
    if out.exists():
        return out.name
    torch.set_num_threads(1)
    if method in BASELINES:
        r = run_stream_arm(stream, seed, **BASELINES[method])
    else:
        r = run_stream_arm(stream, seed, method=method, lam0=lam0)
    row = dict(stream=stream, method=method, lam0=lam0, seed=seed,
               prequential_acc=r["prequential_acc"], n_windows=r["n_windows"])
    payload = dict(row=row, per_window=r["per_window"])
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return out.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    streams = ["electricity", "sea"] if args.quick else STREAMS
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out else root / "results" / "streaming_drift_20260623"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for stream in streams:
        for seed in seeds:
            for lam in GRID:
                cells.append((stream, "fixed", lam, seed))
            cells.append((stream, "ctrl", 0.0, seed))
            for b in BASELINES:
                cells.append((stream, b, 0.0, seed))
    todo = [(*c, str(runs_dir)) for c in cells
            if not (runs_dir / f"{cell_key(*c)}.json").exists()]
    t0 = time.time()
    print(f"plan: {len(cells)} cells, {len(cells)-len(todo)} cached, {len(todo)} to run", flush=True)
    done = 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one_cell, a) for a in todo]
            for fu in as_completed(futs):
                fu.result()
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    if all((runs_dir / f"{cell_key(*c)}.json").exists() for c in cells):
        rows, traces = [], []
        for f in sorted(runs_dir.glob("*.json")):
            pl = json.loads(f.read_text())
            rows.append(pl["row"])
            for w in pl["per_window"]:
                traces.append(dict(stream=pl["row"]["stream"], method=pl["row"]["method"],
                                   lam0=pl["row"]["lam0"], seed=pl["row"]["seed"], **w))
        raw = pd.DataFrame(rows)
        raw.to_csv(out_dir / "streaming_raw.csv", index=False)
        pd.DataFrame(traces).to_csv(out_dir / "streaming_traces.csv", index=False)
        man = dict(generated_utc=utc_now(), python=platform.python_version(),
                   torch=torch.__version__, numpy=np.__version__,
                   scipy=__import__("scipy").__version__, sklearn=__import__("sklearn").__version__,
                   streams=streams, grid=GRID, seeds=seeds, window=WINDOW,
                   inner_epochs=INNER_EPOCHS, lr=LR, n_rows=int(len(raw)),
                   sha256={"streaming_raw.csv": sha256(out_dir / "streaming_raw.csv")})
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"DONE rows={len(raw)} -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print("PARTIAL; re-run to resume.", flush=True)


if __name__ == "__main__":
    main()
