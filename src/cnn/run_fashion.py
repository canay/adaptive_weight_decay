#!/usr/bin/env python3
"""Frozen Fashion-MNIST architecture layer for the F06 benchmark pivot."""
from __future__ import annotations

import argparse
import hashlib
import json
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

RUN_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
FIDELITY_SRC = PROJECT / "experiments" / "2026-08-26_codex_local_strong_baseline_fidelity" / "src"
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(FIDELITY_SRC))
import q1_suite as Q  # noqa: E402
from strong_baseline_fidelity import AdamDecayVariant  # noqa: E402

torch.set_num_threads(1)
DATASETS = ("fashion_full", "fashion_n20")
STRONG = {"awd", "swd", "cwd"}
ARMS = ("none", "fixed1", "ctrlA", "ctrlB", "adadecay", "awd", "swd", "cwd")
FASHION_RAW = PROJECT / "data_cache" / "FashionMNIST" / "raw"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def read_idx_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"truncated IDX image file: {path}")
    magic = int.from_bytes(raw[0:4], "big")
    count = int.from_bytes(raw[4:8], "big")
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    if magic != 2051 or len(raw) != 16 + count * rows * cols:
        raise ValueError(f"invalid IDX image header/size: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols).copy()


def read_idx_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated IDX label file: {path}")
    magic = int.from_bytes(raw[0:4], "big")
    count = int.from_bytes(raw[4:8], "big")
    if magic != 2049 or len(raw) != 8 + count:
        raise ValueError(f"invalid IDX label header/size: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()


def load_fashion_raw(noisy: bool):
    train_images = read_idx_images(FASHION_RAW / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(FASHION_RAW / "train-labels-idx1-ubyte")
    test_images = read_idx_images(FASHION_RAW / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(FASHION_RAW / "t10k-labels-idx1-ubyte")
    if train_images.shape != (60000, 28, 28) or test_images.shape != (10000, 28, 28):
        raise ValueError("unexpected Fashion-MNIST dimensions")
    rng = np.random.RandomState(0)
    n_sub = 3000 if noisy else 8000
    subset = rng.choice(len(train_images), n_sub, replace=False)
    Xtr = train_images[subset].astype(np.float32) / 255.0
    ytr = train_labels[subset].astype(np.int64)
    if noisy:
        noise_rng = np.random.RandomState(123)
        flip = noise_rng.rand(len(ytr)) < 0.2
        ytr = ytr.copy()
        ytr[flip] = (ytr[flip] + noise_rng.randint(1, 10, flip.sum())) % 10
    mean, std = Xtr.mean(), Xtr.std() + 1e-8
    Xtr = ((Xtr - mean) / std)[:, None, :, :].astype(np.float32)
    Xte = ((test_images[:4000].astype(np.float32) / 255.0 - mean) / std)[:, None, :, :].astype(np.float32)
    yte = test_labels[:4000].astype(np.int64)
    ev = np.random.RandomState(1).choice(len(Xtr), min(4096, len(Xtr)), replace=False)
    return Xtr, ytr, Xte, yte, 10, ev.astype(np.int64)


def prepare_cache(out_dir: Path) -> Path:
    cache = out_dir / "dataset_cache"
    manifest = cache / "manifest.json"
    if manifest.exists():
        try:
            current = json.loads(manifest.read_text(encoding="utf-8"))
            if len(current.get("tasks", [])) == 2 and all(
                (cache / item["file"]).exists()
                and file_sha(cache / item["file"]) == item["sha256"]
                for item in current["tasks"]
            ):
                return manifest
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    tasks = []
    for name in DATASETS:
        loaded = load_fashion_raw(name.endswith("n20"))
        path = cache / f"{name}.npz"
        arrays = {
            "Xtr": loaded[0], "ytr": loaded[1],
            "Xte": loaded[2], "yte": loaded[3],
            "n_class": np.array(loaded[4], dtype=np.int64),
            "ev": loaded[5],
        }
        atomic_npz(path, arrays)
        tasks.append({
            "task": name, "file": path.name, "sha256": file_sha(path),
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        })
        print(f"DATA_CACHE {name} sha={tasks[-1]['sha256']}", flush=True)
    atomic_json(manifest, {
        "schema_version": 1, "dataset": "Fashion-MNIST",
        "subset_seed": 0, "noise_seed": 123, "noise_fraction": 0.2,
        "test_prefix_count": 4000, "tasks": tasks,
    })
    return manifest


def load_frozen(cache_dir: Path, name: str):
    data = np.load(cache_dir / f"{name}.npz")
    return (
        torch.tensor(data["Xtr"], dtype=torch.float32),
        torch.tensor(data["ytr"], dtype=torch.long),
        torch.tensor(data["Xte"], dtype=torch.float32),
        torch.tensor(data["yte"], dtype=torch.long),
        int(data["n_class"]),
        torch.tensor(data["ev"], dtype=torch.long),
    )


def train_strong(name, seed, method, data, epoch_cap=0):
    kind, planned_epochs, batch, _ = Q.dataset_spec(name)
    epochs = min(planned_epochs, epoch_cap) if epoch_cap else planned_epochs
    Xtr, ytr, Xte, yte, n_class, ev = data
    model = Q.make_model(kind, Xtr, n_class, seed)
    opt = AdamDecayVariant(model.parameters(), method=method, lr=Q.LR)
    lossf = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(60000 + seed)
    traces = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=generator)
        losses, steps = [], []
        for offset in range(0, len(perm), batch):
            idx = perm[offset:offset + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                return {"diverged": True, "hist": traces}
            loss.backward()
            steps.append(opt.step())
            losses.append(float(loss.detach()))
        traces.append({
            "epoch": epoch, "train_loss_batch_mean": float(np.mean(losses)),
            "weight_norm": Q.C.param_norm(model),
            "effective_decay_mean": float(np.mean([step["effective_decay"] for step in steps])),
            "gradient_norm_mean": float(np.mean([step["grad_norm"] for step in steps])),
            "vbar_sqrt_mean": float(np.mean([step["vbar_sqrt"] for step in steps])),
            "cwd_mask_fraction_mean": (
                float(np.mean([step["cwd_mask_fraction"] for step in steps]))
                if method == "cwd" else None
            ),
        })
    train = Q.C.evaluate(model, Xtr[ev], ytr[ev])
    test = Q.C.evaluate(model, Xte, yte)
    return {
        "diverged": False, "test_acc": test["acc"], "test_ece": test["ece"],
        "train_acc": train["acc"], "gap": train["acc"] - test["acc"],
        "hist": traces, "epoch_count": epochs,
    }


def execute(name, seed, arm, data, epoch_cap):
    if arm in STRONG:
        return train_strong(name, seed, arm, data, epoch_cap)
    if epoch_cap:
        raise ValueError("epoch-cap smoke is restricted to strong baselines")
    if arm == "none":
        return Q.train_arm(name, seed, "fixed", 0.0, _data=data)
    if arm == "fixed1":
        return Q.train_arm(name, seed, "fixed", 1.0, _data=data)
    if arm == "adadecay":
        return Q.train_arm(name, seed, "adadecay_regime", 0.0, _data=data)
    return Q.train_arm(name, seed, arm, 0.0, _data=data)


def valid(path, expected):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete" and all(payload.get(k) == value for k, value in expected.items())


def run_cell(args):
    name, arm, seed, out_dir, cache_dir, root, epoch_cap = args
    key = f"{name}__{arm}__s{seed}"
    path = Path(out_dir) / "runs" / f"{key}.json"
    expected = {"dataset": name, "arm": arm, "seed": seed,
                "epoch_cap": epoch_cap, "evidence_root_sha256": root}
    if path.exists() and valid(path, expected):
        return str(path)
    started = time.time()
    result = execute(name, seed, arm, load_frozen(Path(cache_dir), name), epoch_cap)
    if result.get("diverged"):
        raise FloatingPointError(key)
    row = {"dataset": name, "arm": arm, "seed": seed,
           "test_acc": float(result["test_acc"]), "test_ece": float(result["test_ece"]),
           "train_acc": float(result["train_acc"]), "gap": float(result["gap"]),
           "runtime_seconds": time.time() - started}
    atomic_json(path, {**expected, "status": "complete", "completed_utc": utc_now(),
                       "row": row, "hist": result.get("hist", [])})
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epoch-cap", type=int, default=0)
    args = parser.parse_args()
    arms = args.arms.split(",")
    seeds = [int(value) for value in args.seeds.split(",")]
    if any(arm not in ARMS for arm in arms):
        raise ValueError("unknown arm")
    if args.epoch_cap and any(arm not in STRONG for arm in arms):
        raise ValueError("smoke with epoch cap permits only strong baselines")
    out_dir = Path(args.out).resolve()
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    cache_manifest = prepare_cache(out_dir)
    hashes = {
        "runner": file_sha(Path(__file__)),
        "strong_optimizer": file_sha(FIDELITY_SRC / "strong_baseline_fidelity.py"),
        "q1_suite": file_sha(PROJECT / "code" / "q1_suite.py"),
        "common6": file_sha(PROJECT / "code" / "common6.py"),
        "design_lock": file_sha(PROJECT / "MD" / "02_design" / "DESIGN_LOCK_20260826.md"),
        "dataset_cache_manifest": file_sha(cache_manifest),
        "fashion_train_images": file_sha(FASHION_RAW / "train-images-idx3-ubyte"),
        "fashion_train_labels": file_sha(FASHION_RAW / "train-labels-idx1-ubyte"),
        "fashion_test_images": file_sha(FASHION_RAW / "t10k-images-idx3-ubyte"),
        "fashion_test_labels": file_sha(FASHION_RAW / "t10k-labels-idx1-ubyte"),
    }
    root = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest().upper()
    atomic_json(out_dir / "source_bundle.json", {"generated_utc": utc_now(), "hashes": hashes,
                                                   "evidence_root_sha256": root})
    plan = [(name, arm, seed) for name in DATASETS for arm in arms for seed in seeds]
    todo, complete = [], []
    for name, arm, seed in plan:
        path = out_dir / "runs" / f"{name}__{arm}__s{seed}.json"
        expected = {"dataset": name, "arm": arm, "seed": seed,
                    "epoch_cap": args.epoch_cap, "evidence_root_sha256": root}
        if path.exists() and valid(path, expected):
            complete.append(path)
        else:
            todo.append((name, arm, seed, str(out_dir), str(cache_manifest.parent), root, args.epoch_cap))
    print(f"PLAN total={len(plan)} valid={len(complete)} todo={len(todo)}", flush=True)
    started = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_cell, item) for item in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                path = Path(future.result())
                complete.append(path)
                atomic_json(out_dir / "progress.json", {
                    "utc": utc_now(), "planned_cells": len(plan),
                    "completed_cells": len(complete), "latest_cell": path.name,
                    "elapsed_seconds": time.time() - started, "evidence_root_sha256": root,
                })
                print(f"PROGRESS {index}/{len(todo)} complete={len(complete)}/{len(plan)} elapsed={time.time()-started:.1f}s", flush=True)
    if len(complete) != len(plan):
        print("PARTIAL", flush=True)
        return
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in complete]
    rows = [payload["row"] for payload in payloads]
    pd.DataFrame(rows).sort_values(["dataset", "arm", "seed"]).to_csv(out_dir / "architecture_raw.csv", index=False)
    atomic_json(out_dir / "summary.json", {
        "status": "complete", "generated_utc": utc_now(), "cell_count": len(rows),
        "datasets": list(DATASETS), "arms": arms, "seeds": seeds,
        "epoch_cap": args.epoch_cap, "evidence_root_sha256": root,
        "python": platform.python_version(), "torch": torch.__version__,
    })
    print(f"DONE cells={len(rows)} root={root}", flush=True)


if __name__ == "__main__":
    main()
