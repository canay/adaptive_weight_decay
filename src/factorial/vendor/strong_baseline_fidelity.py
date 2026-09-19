#!/usr/bin/env python3
"""F06 strong-baseline fidelity smoke.

Operation: F06-NOVELTY-ACCEPTABILITY-UPGRADE-20260826-01

Implements three close-prior update mechanisms in an isolated lineage:
AWD (Ghiasi et al., NeurIPS 2023, Eq. 7; adapted to Adam's decoupled update),
SWD/AdamS (Xie et al., NeurIPS 2023, Algorithm 2), and
CWD (Chen et al., ICLR 2026, Algorithm 1).

Each dataset-method-seed cell is atomic and resume-validated. This script is a
fidelity/durability smoke, not confirmatory evidence.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RUN_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "code"))
import common6 as C  # noqa: E402

torch.set_num_threads(1)
LR = 3e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
METHODS = ("adamw", "awd", "swd", "cwd")
DEFAULTS = {
    "adamw": {"weight_decay": 1.0},
    "awd": {"alpha_awd": 0.022},
    "swd": {"weight_decay": 5e-4},
    "cwd": {"weight_decay": 1.0},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class AdamDecayVariant:
    """Reference implementation with traceable scalar update diagnostics."""

    def __init__(self, params, method: str, lr: float = LR):
        if method not in METHODS:
            raise ValueError(method)
        self.params = list(params)
        self.method = method
        self.lr = float(lr)
        self.beta1, self.beta2 = BETAS
        self.eps = EPS
        self.step_index = 0
        self.state = {
            id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            for p in self.params
        }
        self.awd_ema = 0.0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> dict:
        active = [p for p in self.params if p.grad is not None]
        if not active:
            raise RuntimeError("no gradients")
        self.step_index += 1
        grad_sq = sum(float(p.grad.detach().pow(2).sum()) for p in active)
        param_sq = sum(float(p.detach().pow(2).sum()) for p in active)
        grad_norm = math.sqrt(grad_sq)
        param_norm = math.sqrt(param_sq)

        vhats = []
        updates = []
        bc1 = 1.0 - self.beta1 ** self.step_index
        bc2 = 1.0 - self.beta2 ** self.step_index
        for p in active:
            st = self.state[id(p)]
            g = p.grad.detach()
            st["m"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            st["v"].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            mhat = st["m"] / bc1
            vhat = st["v"] / bc2
            update = mhat / (vhat.sqrt() + self.eps)
            vhats.append(vhat)
            updates.append(update)

        vbar_sqrt = math.sqrt(
            sum(float(v.sum()) for v in vhats) /
            max(1, sum(v.numel() for v in vhats))
        )
        mask_count = 0
        elem_count = 0

        if self.method == "awd":
            raw = DEFAULTS["awd"]["alpha_awd"] * grad_norm / max(param_norm, 1e-12)
            self.awd_ema = 0.1 * self.awd_ema + 0.9 * raw
            effective_decay = self.awd_ema
        elif self.method == "swd":
            effective_decay = DEFAULTS["swd"]["weight_decay"] / max(vbar_sqrt, 1e-12)
        else:
            effective_decay = DEFAULTS[self.method]["weight_decay"]

        for p, update in zip(active, updates):
            if self.method == "cwd":
                mask = (update * p >= 0).to(p.dtype)
                mask_count += int(mask.sum())
                elem_count += mask.numel()
                p.add_(update + effective_decay * mask * p, alpha=-self.lr)
            else:
                p.add_(update + effective_decay * p, alpha=-self.lr)

        return {
            "effective_decay": float(effective_decay),
            "grad_norm": grad_norm,
            "param_norm_pre": param_norm,
            "vbar_sqrt": vbar_sqrt,
            "cwd_mask_fraction": (
                float(mask_count) / elem_count if self.method == "cwd" else None
            ),
        }


def self_test() -> dict:
    results = {}
    for method in METHODS:
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        p.grad = torch.tensor([0.5, 0.25])
        opt = AdamDecayVariant([p], method=method, lr=0.01)
        trace = opt.step()
        if not torch.isfinite(p).all():
            raise AssertionError(f"{method}: non-finite first step")
        if method == "cwd" and not (0.0 < trace["cwd_mask_fraction"] < 1.0):
            raise AssertionError("cwd mask did not exercise both branches")
        if method == "awd" and trace["effective_decay"] <= 0:
            raise AssertionError("awd decay must be positive")
        if method == "swd" and trace["vbar_sqrt"] <= 0:
            raise AssertionError("swd vbar must be positive")
        results[method] = {"parameter_after": p.tolist(), **trace}
    return results


def load_dataset(name: str):
    if name not in ("digits_full", "p_digits500_n20"):
        raise ValueError(f"fidelity smoke dataset not allowed: {name}")
    return C.load_task(name)


def valid_cached(path: Path, expected: dict) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("status") == "complete"
        and all(data.get(k) == v for k, v in expected.items())
        and isinstance(data.get("epochs"), list)
        and len(data["epochs"]) == expected["epoch_count"]
    )


def run_cell(dataset: str, method: str, seed: int, epoch_count: int,
             out_dir: Path, source_hash: str) -> dict:
    key = f"{dataset}__{method}__s{seed}__e{epoch_count}"
    out = out_dir / "runs" / f"{key}.json"
    expected = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "epoch_count": epoch_count,
        "source_sha256": source_hash,
    }
    if out.exists() and valid_cached(out, expected):
        return {"key": key, "cached": True, "path": str(out)}

    data = load_dataset(dataset)
    model = C.make_mlp(data["Xtr"].shape[1], data["n_class"], seed)
    opt = AdamDecayVariant(model.parameters(), method=method)
    lossf = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(60000 + seed)
    batch = 128
    traces = []
    started = time.time()
    for epoch in range(epoch_count):
        model.train()
        perm = torch.randperm(len(data["Xtr"]), generator=generator)
        losses = []
        step_traces = []
        for offset in range(0, len(perm), batch):
            idx = perm[offset:offset + batch]
            opt.zero_grad()
            loss = lossf(model(data["Xtr"][idx]), data["ytr"][idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{key}: non-finite loss")
            loss.backward()
            step_traces.append(opt.step())
            losses.append(float(loss.detach()))
        train = C.evaluate(model, data["Xtr"][data["tr_eval_idx"]],
                           data["ytr"][data["tr_eval_idx"]])
        traces.append({
            "epoch": epoch,
            "train_loss_batch_mean": float(np.mean(losses)),
            "train_accuracy": train["acc"],
            "weight_norm": C.param_norm(model),
            "effective_decay_mean": float(np.mean(
                [s["effective_decay"] for s in step_traces]
            )),
            "gradient_norm_mean": float(np.mean(
                [s["grad_norm"] for s in step_traces]
            )),
            "vbar_sqrt_mean": float(np.mean(
                [s["vbar_sqrt"] for s in step_traces]
            )),
            "cwd_mask_fraction_mean": (
                float(np.mean([s["cwd_mask_fraction"] for s in step_traces]))
                if method == "cwd" else None
            ),
        })
        atomic_json(out_dir / "heartbeat.json", {
            "utc": utc_now(), "cell": key, "epoch_completed": epoch,
            "epoch_count": epoch_count, "source_sha256": source_hash,
        })

    train = C.evaluate(model, data["Xtr"][data["tr_eval_idx"]],
                       data["ytr"][data["tr_eval_idx"]])
    test = C.evaluate(model, data["Xte"], data["yte"])
    payload = {
        **expected,
        "status": "complete",
        "completed_utc": utc_now(),
        "defaults": DEFAULTS[method],
        "runtime_seconds": time.time() - started,
        "train": train,
        "test": test,
        "gap": train["acc"] - test["acc"],
        "epochs": traces,
    }
    atomic_json(out, payload)
    return {"key": key, "cached": False, "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default="digits_full,p_digits500_n20")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stop-after-cells", type=int, default=0)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    source_hash = sha256(Path(__file__))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = self_test()
    atomic_json(out_dir / "self_test.json", {
        "status": "PASS", "utc": utc_now(), "source_sha256": source_hash,
        "tests": tests,
    })
    if args.self_test_only:
        print("SELF_TEST=PASS", flush=True)
        return

    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]
    for method in methods:
        if method not in METHODS:
            raise ValueError(method)
    plan = [(d, m, s) for d in datasets for m in methods for s in seeds]
    completed = []
    new_count = 0
    for dataset, method, seed in plan:
        item = run_cell(dataset, method, seed, args.epochs, out_dir, source_hash)
        completed.append(item)
        if not item["cached"]:
            new_count += 1
        atomic_json(out_dir / "progress.json", {
            "utc": utc_now(), "planned_cells": len(plan),
            "completed_cells": len(completed), "items": completed,
            "source_sha256": source_hash,
        })
        print(f"{len(completed)}/{len(plan)} {item['key']} cached={item['cached']}", flush=True)
        if args.stop_after_cells and new_count >= args.stop_after_cells:
            print("INTENTIONAL_STOP_BETWEEN_ATOMIC_CELLS", flush=True)
            return

    payloads = [json.loads(Path(item["path"]).read_text(encoding="utf-8"))
                for item in completed]
    atomic_json(out_dir / "summary.json", {
        "status": "complete",
        "generated_utc": utc_now(),
        "source_sha256": source_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cell_count": len(payloads),
        "cells": [{
            "dataset": p["dataset"], "method": p["method"],
            "seed": p["seed"], "test_acc": p["test"]["acc"],
            "test_ece": p["test"]["ece"], "gap": p["gap"],
            "runtime_seconds": p["runtime_seconds"],
        } for p in payloads],
    })
    print(f"DONE cells={len(payloads)} source={source_hash}", flush=True)


if __name__ == "__main__":
    main()

