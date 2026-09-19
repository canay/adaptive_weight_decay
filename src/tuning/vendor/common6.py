"""Paper 6: validation-free internal-signal controller for AdamW weight decay.

Tasks: adult / bank_marketing (cached npz) / sklearn digits, full and reduced
(overfitting-prone) variants. Model: MLP d-64-32-C. CPU, 1 thread per worker.
"""
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = ROOT / "data"
SHARED_TABULAR_DATA_DIR = ROOT.parent / "SCI-f01-explanation_reliability" / "data"
SHARED_DIGITS_NPZ = ROOT.parent / "SCI-f05-optimizer_atlas" / "data" / "digits.npz"
DATA_DIR = str(LOCAL_DATA_DIR if (LOCAL_DATA_DIR / "adult.npz").exists() else SHARED_TABULAR_DATA_DIR)
DIGITS_NPZ = str((LOCAL_DATA_DIR / "digits.npz") if (LOCAL_DATA_DIR / "digits.npz").exists() else SHARED_DIGITS_NPZ)

GRID = [0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
BATCH = 256
HIDDEN = [64, 32]

TASKS = {
    "adult_full":  dict(src="adult",  n_sub=None, epochs=20),
    "adult_2k":    dict(src="adult",  n_sub=2000, epochs=30),
    "bank_full":   dict(src="bank",   n_sub=None, epochs=20),
    "bank_2k":     dict(src="bank",   n_sub=2000, epochs=30),
    "digits_full": dict(src="digits", n_sub=None, epochs=30),
    "digits_500":  dict(src="digits", n_sub=500,  epochs=30),
}
# probe variants (design pilot only)
for _n, _cfg in [
    ("p_adult2k_b128", dict(src="adult", n_sub=2000, epochs=40, batch=128)),
    ("p_adult2k_n20", dict(src="adult", n_sub=2000, epochs=40, batch=128,
                           noise=0.2)),
    ("p_adult1k_n20", dict(src="adult", n_sub=1000, epochs=40, batch=128,
                           noise=0.2)),
    ("p_digits500_n20", dict(src="digits", n_sub=500, epochs=40, batch=128,
                             noise=0.2)),
    ("p_digits500_b128", dict(src="digits", n_sub=500, epochs=40, batch=128)),
]:
    TASKS[_n] = _cfg

_CACHE = {}


def load_task(name):
    if name in _CACHE:
        return _CACHE[name]
    cfg = TASKS[name]
    rng = np.random.RandomState(0)  # fixed: same data split for all methods
    if cfg["src"] == "digits":
        dd = np.load(DIGITS_NPZ)
        X = dd["X"].astype(np.float64) / 16.0
        y = dd["y"].astype(np.int64)
        idx = rng.permutation(len(X))
        n_te = int(0.3 * len(X))
        te, tr = idx[:n_te], idx[n_te:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    else:
        fn = {"adult": "adult.npz", "bank": "bank_marketing.npz"}[cfg["src"]]
        dd = np.load(os.path.join(DATA_DIR, fn), allow_pickle=True)
        Xtr = np.asarray(dd["X_train"], dtype=np.float64)
        ytr = np.asarray(dd["y_train"]).astype(np.int64)
        Xte = np.asarray(dd["X_test"], dtype=np.float64)
        yte = np.asarray(dd["y_test"]).astype(np.int64)
    if cfg["n_sub"] is not None and len(Xtr) > cfg["n_sub"]:
        sidx = rng.choice(len(Xtr), cfg["n_sub"], replace=False)
        Xtr, ytr = Xtr[sidx], ytr[sidx]
    noise = cfg.get("noise", 0.0)
    if noise > 0:  # symmetric label noise on the TRAINING set only
        nc = int(max(ytr.max(), yte.max()) + 1)
        nrng = np.random.RandomState(123)
        flip = nrng.rand(len(ytr)) < noise
        ytr = ytr.copy()
        ytr[flip] = (ytr[flip] + nrng.randint(1, nc, flip.sum())) % nc
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8

    def t(a):
        return torch.tensor((a - mu) / sd, dtype=torch.float32)

    out = dict(Xtr=t(Xtr), ytr=torch.tensor(ytr), Xte=t(Xte),
               yte=torch.tensor(yte))
    ne = min(4096, len(Xtr))
    out["tr_eval_idx"] = torch.tensor(
        np.random.RandomState(1).choice(len(Xtr), ne, replace=False))
    out["n_class"] = int(max(ytr.max(), yte.max()) + 1)
    _CACHE[name] = out
    return out


def make_mlp(d_in, n_class, seed):
    torch.manual_seed(40000 + seed)
    layers, prev = [], d_in
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_class))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, y, n_bins=15):
    model.eval()
    logits = model(X)
    if not torch.isfinite(logits).all():
        return dict(acc=float("nan"), loss=float("nan"), ece=float("nan"))
    loss = float(nn.functional.cross_entropy(logits, y))
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    acc_v = (pred == y).float()
    acc = float(acc_v.mean())
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += float(m.sum()) / n * abs(float(acc_v[m].mean()) -
                                            float(conf[m].mean()))
    return dict(acc=acc, loss=loss, ece=float(ece))


@torch.no_grad()
def param_norm(model):
    return float(torch.sqrt(sum(p.pow(2).sum() for p in model.parameters())))


def run_unit(u):
    """u: dict(task, method ['fixed'|'ctrl'], lam0, lr, seed,
               rule, eta, beta, warmup [ctrl only])."""
    task, lr, seed = u["task"], float(u["lr"]), int(u["seed"])
    lam0 = float(u["lam0"])
    is_ctrl = u["method"] == "ctrl"
    rule = u.get("rule", "A")
    eta = float(u.get("eta", 0.3))
    beta = float(u.get("beta", 0.5))
    warmup = int(u.get("warmup", 2))
    cfg = TASKS[task]
    D = load_task(task)
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    n = len(Xtr)
    model = make_mlp(Xtr.shape[1], D["n_class"], seed)
    lam = lam0
    if is_ctrl and rule in ("B", "C") and lam < 1e-6:
        lam = 1e-6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=lam)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(60000 + seed)
    hist = []
    wn_prev = param_norm(model)
    gbar = 0.0
    diverged = False
    batch = cfg.get("batch", BATCH)
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        losses, ur_acc = [], []
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
            record_ur = (n_steps % 5 == 0)
            if record_ur:
                prev = [p.detach().clone() for p in model.parameters()]
            opt.step()
            if record_ur:
                with torch.no_grad():
                    dn = float(torch.sqrt(sum((p.detach() - q).pow(2).sum()
                               for p, q in zip(model.parameters(), prev))))
                    wn = float(torch.sqrt(sum(p.detach().pow(2).sum()
                               for p in model.parameters())))
                ur_acc.append(dn / (wn + 1e-12))
            losses.append(lv)
            n_steps += 1
        if diverged:
            break
        wn = param_norm(model)
        gr = (wn - wn_prev) / (wn_prev + 1e-12)
        wn_prev = wn
        gbar = beta * gbar + (1 - beta) * gr
        s = gbar / (lr * n_steps)
        ei = D["tr_eval_idx"]
        etr = evaluate(model, Xtr[ei], ytr[ei])
        ete = evaluate(model, Xte, yte)
        hist.append(dict(ep=ep, lam=lam, train_loss=float(np.mean(losses)),
                         train_acc=etr["acc"], test_acc=ete["acc"],
                         test_loss=ete["loss"], gap=etr["acc"] - ete["acc"],
                         wnorm=wn, g=gr, gbar=gbar, s=s,
                         upd_ratio=float(np.mean(ur_acc))))
        if is_ctrl and ep + 1 >= warmup and ep + 1 < cfg["epochs"]:
            if rule == "A":
                lam = min(max(lam + eta * min(max(s, -1.0), 1.0), 0.0), 30.0)
            elif rule == "B":
                lam = min(max(lam * math.exp(eta * math.copysign(1.0, gbar)),
                              1e-6), 30.0)
            elif rule == "C":
                lam = min(max(lam * math.exp(eta * min(max(s, -1.0), 1.0)),
                              1e-6), 30.0)
            opt.param_groups[0]["weight_decay"] = lam
    if diverged or not hist:
        return dict(diverged=True, hist=hist)
    h = hist[-1]
    ei = D["tr_eval_idx"]
    etr = evaluate(model, Xtr[ei], ytr[ei])
    ete = evaluate(model, Xte, yte)
    return dict(diverged=False, test_acc=ete["acc"], test_loss=ete["loss"],
                test_ece=ete["ece"], train_acc=etr["acc"],
                train_loss=etr["loss"], gap=etr["acc"] - ete["acc"],
                lam_final=h["lam"],
                lam_last5=float(np.mean([x["lam"] for x in hist[-5:]])),
                hist=hist)
