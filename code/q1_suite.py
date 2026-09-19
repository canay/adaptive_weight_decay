#!/usr/bin/env python3
"""Unified Q1 experiment suite: improved controller across many datasets + a CNN.

Controllers (rule A update lambda <- clip[0,30](lambda + eta*clip[-1,1](s))):
  ctrlA  norm-growth signal  s = EMA(rel weight-norm growth)/(lr*K)      [original]
  ctrlB  loss-progress signal s = 10 * relative train-loss decrease       [improved primary]
  ctrlBpp loss-progress + norm-collapse release (two-sided recovery)

Datasets:
  tabular (MLP):  adult, bank, digits + 6 OpenML sets (phoneme, spambase, qsar-biodeg,
                  waveform-5000, mfeat-karhunen, satimage), each full and a noisy-small (_n20) variant
  image (CNN):    Fashion-MNIST subset, full and noisy-small

Arms: fixed-decay grid, ctrlA, ctrlB, AdaDecay (default + regime). The budget
(validation-tuned) comparison is in run_budget.py / computed here via method 'valtuned'.

Resumable (per-cell JSON) and parallel. Usage:
  python code/q1_suite.py            # full
  python code/q1_suite.py --quick    # tiny smoke
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common6 as C  # noqa: E402
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
LR = 3e-3
GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
IMG_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]

C.TASKS["adult_n20"] = dict(src="adult", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["bank_n20"] = dict(src="bank", n_sub=2000, epochs=40, batch=128, noise=0.2)
C.TASKS["digits_n20"] = dict(src="digits", n_sub=500, epochs=40, batch=128, noise=0.2)

# OpenML tabular datasets: name -> (openml_name, full_epochs)
OPENML = {
    "phoneme": "phoneme", "spambase": "spambase", "qsar": "qsar-biodeg",
    "waveform": "waveform-5000", "mfeat": "mfeat-karhunen", "satimage": "satimage",
}
_CACHE = {}


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def load_openml_task(oname, n_sub, noise):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(oname, version=1, as_frame=False)
    X = np.asarray(d.data, dtype=np.float64)
    classes = sorted(set(np.asarray(d.target).tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in np.asarray(d.target).tolist()], dtype=np.int64)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    nte = int(0.3 * len(X))
    te, tr = idx[:nte], idx[nte:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    if n_sub and len(Xtr) > n_sub:
        s = rng.choice(len(Xtr), n_sub, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    if noise > 0:
        nc = int(max(ytr.max(), yte.max()) + 1)
        nr = np.random.RandomState(123)
        flip = nr.rand(len(ytr)) < noise
        ytr = ytr.copy()
        ytr[flip] = (ytr[flip] + nr.randint(1, nc, flip.sum())) % nc
    Xtr, Xte = _standardize(Xtr, Xte)
    n = len(Xtr)
    ev = np.random.RandomState(1).choice(n, min(4096, n), replace=False)
    return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr),
            torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte),
            int(max(ytr.max(), yte.max()) + 1), torch.tensor(ev))


def load_fashion(noisy):
    from torchvision import datasets
    root = str(ROOT / "data_cache")
    tr = datasets.FashionMNIST(root, train=True, download=True)
    te = datasets.FashionMNIST(root, train=False, download=True)
    rng = np.random.RandomState(0)
    Xtr = tr.data.float() / 255.0
    ytr = tr.targets.clone()
    n_sub = 3000 if noisy else 8000
    sidx = rng.choice(len(Xtr), n_sub, replace=False)
    Xtr, ytr = Xtr[sidx], ytr[sidx]
    if noisy:
        nr = np.random.RandomState(123)
        flip = nr.rand(len(ytr)) < 0.2
        ytr[torch.tensor(flip)] = torch.tensor(
            (ytr.numpy()[flip] + nr.randint(1, 10, flip.sum())) % 10)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    Xtr = ((Xtr - mu) / sd).unsqueeze(1)
    Xte = ((te.data.float() / 255.0 - mu) / sd).unsqueeze(1)[:4000]
    yte = te.targets[:4000]
    ev = torch.tensor(np.random.RandomState(1).choice(len(Xtr), min(4096, len(Xtr)), replace=False))
    return Xtr, ytr, Xte, yte, 10, ev


def dataset_spec(name):
    """Return (kind, epochs, batch, grid)."""
    if name.startswith("fashion"):
        return "cnn", (40 if name.endswith("n20") else 20), 128, IMG_GRID
    if name.endswith("_n20"):
        return "mlp", 40, 128, GRID
    return "mlp", 30, 256, GRID


def load_dataset(name):
    if name in _CACHE:
        return _CACHE[name]
    if name.startswith("fashion"):
        out = load_fashion(name.endswith("n20"))
    elif name in ("adult_full", "bank_full", "digits_full", "adult_n20", "bank_n20", "digits_n20"):
        D = C.load_task(name)
        out = (D["Xtr"], D["ytr"], D["Xte"], D["yte"], D["n_class"], D["tr_eval_idx"])
    else:
        base = name.replace("_full", "").replace("_n20", "")
        oname = OPENML[base]
        if name.endswith("_n20"):
            out = load_openml_task(oname, 2000, 0.2)
        else:
            out = load_openml_task(oname, None, 0.0)
    _CACHE[name] = out
    return out


def make_model(kind, sample_x, n_class, seed):
    torch.manual_seed(40000 + seed)
    if kind == "mlp":
        return C.make_mlp(sample_x.shape[1], n_class, seed)
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_class))


def _adamw(params, lam):
    return torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=lam)


def train_arm(name, seed, method, lam0=0.0, _data=None):
    kind, epochs, batch, _ = dataset_spec(name)
    Xtr, ytr, Xte, yte, n_class, ev = _data if _data is not None else load_dataset(name)
    model = make_model(kind, Xtr, n_class, seed)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(60000 + seed)
    n = len(Xtr)
    is_ad = method.startswith("adadecay")
    is_ap = method == "adamp"
    is_ctrl = method in ("ctrlA", "ctrlB", "ctrlBpp", "ctrlC")
    is_sched = method in ("cosine", "early")
    lam = 0.0 if (is_ad or is_ap) else float(lam0)
    if is_ap:
        from adamp import AdamP
        opt = AdamP(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                    delta=0.1, wd_ratio=0.1)
    else:
        opt = _adamw(model.parameters(), lam)
    alpha = 4.0
    lam_base = 5e-4 if method == "adadecay_default" else 1.0
    lam_max = 1.0
    wn_prev = C.param_norm(model)
    init_norm = wn_prev
    gbar = 0.0
    prev_loss = None
    hist = []
    for ep in range(epochs):
        model.train()
        if method == "cosine":
            lam = lam_max * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs - 1)))
            opt.param_groups[0]["weight_decay"] = lam
        elif method == "early":
            lam = lam_max if ep < epochs // 2 else 0.0
            opt.param_groups[0]["weight_decay"] = lam
        perm = torch.randperm(n, generator=g)
        losses, nsteps = [], 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), ytr[idx])
            lv = float(loss.detach())
            if not math.isfinite(lv):
                return dict(diverged=True, hist=hist)
            loss.backward()
            if is_ad:
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
            losses.append(lv)
            nsteps += 1
        wn = C.param_norm(model)
        gr = (wn - wn_prev) / (wn_prev + 1e-12)
        wn_prev = wn
        gbar = 0.5 * gbar + 0.5 * gr
        tl = float(np.mean(losses))
        rd = 0.0 if prev_loss is None else (prev_loss - tl) / (prev_loss + 1e-8)
        prev_loss = tl
        if is_ctrl and ep + 1 >= 2 and ep + 1 < epochs:
            s_ng = gbar / (LR * max(1, nsteps))
            s_lp = 10.0 * rd
            if method == "ctrlA":
                s = s_ng
            elif method == "ctrlB":
                s = s_lp
            elif method == "ctrlBpp":
                s = -1.0 if (wn < 0.5 * init_norm and rd < 0.01) else s_lp
            else:  # ctrlC: norm-growth + loss-progress gated by memorization dynamics
                tr_acc = C.evaluate(model, Xtr[ev], ytr[ev])["acc"]
                s = s_ng + (s_lp if (tr_acc > 0.95 and rd > 0.01) else 0.0)
            lam = min(max(lam + 1.0 * min(max(s, -1.0), 1.0), 0.0), 30.0)
            opt.param_groups[0]["weight_decay"] = lam
        hist.append(dict(ep=ep, lam=float(lam)))
    etr = C.evaluate(model, Xtr[ev], ytr[ev])
    ete = C.evaluate(model, Xte, yte)
    return dict(diverged=False, test_acc=ete["acc"], test_ece=ete["ece"],
                train_acc=etr["acc"], gap=etr["acc"] - ete["acc"], lam_final=lam, hist=hist)


def val_tuned(name, seed):
    kind, epochs, batch, grid = dataset_spec(name)
    Xtr, ytr, Xte, yte, n_class, ev = load_dataset(name)
    rng = np.random.RandomState(777 + seed)
    idx = rng.permutation(len(Xtr))
    nval = int(0.2 * len(Xtr))
    vi, ti = idx[:nval], idx[nval:]

    def train_fixed(Xa, ya, Xb, yb, lam):
        model = make_model(kind, Xtr, n_class, seed)
        opt = _adamw(model.parameters(), lam)
        lossf = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(60000 + seed)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(Xa), generator=g)
            for i in range(0, len(Xa), batch):
                j = perm[i:i + batch]
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(Xa[j]), ya[j])
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
        return C.evaluate(model, Xb, yb)["acc"]

    best_lam, best = None, -1
    for lam in grid:
        a = train_fixed(Xtr[ti], ytr[ti], Xtr[vi], ytr[vi], lam)
        if a > best:
            best, best_lam = a, lam
    test_acc = train_fixed(Xtr, ytr, Xte, yte, best_lam)
    return dict(diverged=False, test_acc=test_acc, best_lam=best_lam, n_runs=len(grid) + 1)


def arms_for(name):
    _, _, _, grid = dataset_spec(name)
    base = [("fixed", l) for l in grid] + [("ctrlA", 0.0), ("ctrlB", 0.0)]
    if name.startswith("fashion"):
        return base + [("adadecay_regime", 0.0)]
    return base + [("ctrlBpp", 0.0), ("adadecay_regime", 0.0), ("valtuned", 0.0)]


TAB = ["adult", "bank", "digits", "phoneme", "spambase", "qsar", "waveform", "mfeat", "satimage"]
DATASETS = [f"{d}_full" for d in TAB] + [f"{d}_n20" for d in TAB] + ["fashion_full", "fashion_n20"]


def cell_key(name, method, lam0, seed):
    return f"{name}__{method}__l{lam0}__s{seed}".replace(".", "p").replace("-", "m")


def run_cell(args):
    name, method, lam0, seed, runs_dir = args
    out = Path(runs_dir) / f"{cell_key(name, method, lam0, seed)}.json"
    if out.exists():
        return out.name
    torch.set_num_threads(1)
    if method == "valtuned":
        r = val_tuned(name, seed)
    else:
        r = train_arm(name, seed, method, lam0)
    row = dict(name=name, method=method, lam0=lam0, seed=seed, diverged=bool(r.get("diverged", False)))
    for k in ("test_acc", "test_ece", "train_acc", "gap", "lam_final", "best_lam", "n_runs"):
        if k in r:
            row[k] = r[k]
    payload = dict(row=row, hist=r.get("hist", []))
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return out.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [0] if args.quick else ([int(s) for s in args.seeds.split(",")] if args.seeds else [0, 1, 2, 3, 4])
    datasets = (args.datasets.split(",") if args.datasets else
                (["phoneme_full", "fashion_full"] if args.quick else DATASETS))
    out_dir = Path(args.out) if args.out else ROOT / "results" / "q1_suite_20260625"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for name in datasets:
        img = name.startswith("fashion")
        sds = ([0, 1, 2] if img else seeds)
        for method, lam0 in arms_for(name):
            for sd in sds:
                cells.append((name, method, lam0, sd))
    todo = [(*c, str(runs_dir)) for c in cells if not (runs_dir / f"{cell_key(*c)}.json").exists()]
    t0 = time.time()
    print(f"plan: {len(cells)} cells, {len(cells)-len(todo)} cached, {len(todo)} to run", flush=True)
    done = 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_cell, a) for a in todo]
            for fu in as_completed(futs):
                fu.result()
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    if all((runs_dir / f"{cell_key(*c)}.json").exists() for c in cells):
        rows, traces = [], []
        for f in sorted(runs_dir.glob("*.json")):
            pl = json.loads(f.read_text())
            rows.append(pl["row"])
            for h in pl["hist"]:
                traces.append(dict(name=pl["row"]["name"], method=pl["row"]["method"],
                                   lam0=pl["row"]["lam0"], seed=pl["row"]["seed"], **h))
        pd.DataFrame(rows).to_csv(out_dir / "q1_raw.csv", index=False)
        pd.DataFrame(traces).to_csv(out_dir / "q1_traces.csv", index=False)
        man = dict(generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   python=platform.python_version(), torch=torch.__version__,
                   numpy=np.__version__, scipy=__import__("scipy").__version__,
                   sklearn=__import__("sklearn").__version__, datasets=datasets, seeds=seeds,
                   grid=GRID, img_grid=IMG_GRID, n_rows=len(rows))
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"DONE rows={len(rows)} -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print("PARTIAL; re-run to resume.", flush=True)


if __name__ == "__main__":
    main()
