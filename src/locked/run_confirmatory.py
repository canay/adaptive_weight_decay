#!/usr/bin/env python3
"""Locked tabular confirmatory matrix for the F06 scientific rebaseline."""
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

RUN_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
FIDELITY_SRC = PROJECT / "experiments" / "2026-08-26_codex_local_strong_baseline_fidelity" / "src"
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(FIDELITY_SRC))
import q1_suite as Q  # noqa: E402
from strong_baseline_fidelity import AdamDecayVariant  # noqa: E402

torch.set_num_threads(1)
CANONICAL_DATA = PROJECT / "replication_package" / "data"
Q.C.DATA_DIR = str(CANONICAL_DATA)
Q.C.DIGITS_NPZ = str(CANONICAL_DATA / "digits.npz")
STRONG = {"awd", "swd", "cwd"}
ARMS = ("none", "fixed1", "ctrlA", "ctrlB", "ctrlBpp", "adadecay", "awd", "swd", "cwd")
TABULAR = [f"{d}_{regime}" for d in Q.TAB for regime in ("full", "n20")]
OPENML_IDS = {
    "phoneme": 1489, "spambase": 44, "qsar": 1494,
    "waveform": 60, "mfeat": 16, "satimage": 182,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _openml_variant(X, y, noisy: bool):
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    nte = int(0.3 * len(X))
    te, tr = idx[:nte], idx[nte:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    if noisy and len(Xtr) > 2000:
        subset = rng.choice(len(Xtr), 2000, replace=False)
        Xtr, ytr = Xtr[subset], ytr[subset]
    if noisy:
        n_class = int(max(ytr.max(), yte.max()) + 1)
        noise_rng = np.random.RandomState(123)
        flip = noise_rng.rand(len(ytr)) < 0.2
        ytr = ytr.copy()
        ytr[flip] = (ytr[flip] + noise_rng.randint(1, n_class, flip.sum())) % n_class
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xte = ((Xte - mean) / std).astype(np.float32)
    ev = np.random.RandomState(1).choice(len(Xtr), min(4096, len(Xtr)), replace=False)
    return {
        "Xtr": Xtr, "ytr": ytr.astype(np.int64),
        "Xte": Xte, "yte": yte.astype(np.int64),
        "n_class": np.array(int(max(ytr.max(), yte.max()) + 1), dtype=np.int64),
        "ev": ev.astype(np.int64),
    }


def prepare_dataset_cache(out_dir: Path) -> Path:
    cache_dir = out_dir / "dataset_cache"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(
                (cache_dir / item["file"]).exists()
                and file_sha(cache_dir / item["file"]) == item["sha256"]
                for item in current.get("tasks", [])
            ) and len(current.get("tasks", [])) == len(TABULAR):
                return manifest_path
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name in TABULAR:
        path = cache_dir / f"{name}.npz"
        if name.startswith(("adult_", "bank_", "digits_")):
            loaded = Q.load_dataset(name)
            arrays = {
                "Xtr": loaded[0].numpy(), "ytr": loaded[1].numpy(),
                "Xte": loaded[2].numpy(), "yte": loaded[3].numpy(),
                "n_class": np.array(loaded[4], dtype=np.int64),
                "ev": loaded[5].numpy(),
            }
            data_id = {"adult": 1590, "bank": 1461, "digits": "sklearn-load_digits"}[
                name.split("_")[0]
            ]
        else:
            from sklearn.datasets import fetch_openml
            base = name.rsplit("_", 1)[0]
            data_id = OPENML_IDS[base]
            raw = fetch_openml(
                data_id=data_id, as_frame=False,
                data_home=str(out_dir / "openml_raw_cache"),
            )
            X = np.asarray(raw.data, dtype=np.float64)
            classes = sorted(set(np.asarray(raw.target).tolist()))
            class_map = {value: index for index, value in enumerate(classes)}
            y = np.array([class_map[value] for value in np.asarray(raw.target).tolist()], dtype=np.int64)
            arrays = _openml_variant(X, y, name.endswith("_n20"))
        atomic_npz(path, arrays)
        tasks.append({
            "task": name, "file": path.name, "data_id": data_id,
            "sha256": file_sha(path),
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
        })
        print(f"DATA_CACHE {name} id={data_id} sha={tasks[-1]['sha256']}", flush=True)
    atomic_json(manifest_path, {
        "schema_version": 1,
        "split_seed": 0, "evaluation_index_seed": 1,
        "noise_seed": 123, "noise_fraction": 0.2,
        "tasks": tasks,
    })
    return manifest_path


def load_frozen_dataset(cache_dir: Path, name: str):
    data = np.load(cache_dir / f"{name}.npz")
    return (
        torch.tensor(data["Xtr"], dtype=torch.float32),
        torch.tensor(data["ytr"], dtype=torch.long),
        torch.tensor(data["Xte"], dtype=torch.float32),
        torch.tensor(data["yte"], dtype=torch.long),
        int(data["n_class"]),
        torch.tensor(data["ev"], dtype=torch.long),
    )


def source_bundle() -> dict:
    paths = {
        "runner": Path(__file__),
        "strong_optimizer": FIDELITY_SRC / "strong_baseline_fidelity.py",
        "q1_suite": PROJECT / "code" / "q1_suite.py",
        "common6": PROJECT / "code" / "common6.py",
        "design_lock": PROJECT / "MD" / "02_design" / "DESIGN_LOCK_20260826.md",
        "data_manifest": CANONICAL_DATA / "DATA.md",
        "adult_data": CANONICAL_DATA / "adult.npz",
        "bank_data": CANONICAL_DATA / "bank_marketing.npz",
        "digits_data": CANONICAL_DATA / "digits.npz",
    }
    hashes = {key: file_sha(path) for key, path in paths.items()}
    root = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest().upper()
    return {"hashes": hashes, "root_sha256": root}


def train_strong(name: str, seed: int, method: str, frozen_data) -> dict:
    kind, epochs, batch, _ = Q.dataset_spec(name)
    Xtr, ytr, Xte, yte, n_class, ev = frozen_data
    model = Q.make_model(kind, Xtr, n_class, seed)
    opt = AdamDecayVariant(model.parameters(), method=method, lr=Q.LR)
    lossf = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(60000 + seed)
    traces = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=generator)
        losses = []
        steps = []
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
            "epoch": epoch,
            "train_loss_batch_mean": float(np.mean(losses)),
            "weight_norm": Q.C.param_norm(model),
            "effective_decay_mean": float(np.mean([s["effective_decay"] for s in steps])),
            "gradient_norm_mean": float(np.mean([s["grad_norm"] for s in steps])),
            "vbar_sqrt_mean": float(np.mean([s["vbar_sqrt"] for s in steps])),
            "cwd_mask_fraction_mean": (
                float(np.mean([s["cwd_mask_fraction"] for s in steps]))
                if method == "cwd" else None
            ),
        })
    train = Q.C.evaluate(model, Xtr[ev], ytr[ev])
    test = Q.C.evaluate(model, Xte, yte)
    return {
        "diverged": False,
        "test_acc": test["acc"],
        "test_ece": test["ece"],
        "train_acc": train["acc"],
        "gap": train["acc"] - test["acc"],
        "hist": traces,
    }


def execute_arm(name: str, seed: int, arm: str, frozen_data) -> dict:
    if arm in STRONG:
        return train_strong(name, seed, arm, frozen_data)
    if arm == "none":
        return Q.train_arm(name, seed, "fixed", 0.0, _data=frozen_data)
    if arm == "fixed1":
        return Q.train_arm(name, seed, "fixed", 1.0, _data=frozen_data)
    if arm == "adadecay":
        return Q.train_arm(name, seed, "adadecay_regime", 0.0, _data=frozen_data)
    return Q.train_arm(name, seed, arm, 0.0, _data=frozen_data)


def cell_key(dataset: str, arm: str, seed: int) -> str:
    return f"{dataset}__{arm}__s{seed}"


def valid_cell(path: Path, expected: dict) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and all(payload.get(k) == v for k, v in expected.items())
        and isinstance(payload.get("row"), dict)
        and "test_acc" in payload["row"]
    )


def run_cell(args) -> str:
    dataset, arm, seed, out_dir, cache_dir, evidence_root = args
    out = Path(out_dir) / "runs" / f"{cell_key(dataset, arm, seed)}.json"
    expected = {
        "dataset": dataset, "arm": arm, "seed": seed,
        "evidence_root_sha256": evidence_root,
    }
    if out.exists() and valid_cell(out, expected):
        return str(out)
    started = time.time()
    frozen_data = load_frozen_dataset(Path(cache_dir), dataset)
    result = execute_arm(dataset, seed, arm, frozen_data)
    if result.get("diverged"):
        raise FloatingPointError(f"{dataset}/{arm}/s{seed} diverged")
    row = {
        "dataset": dataset, "arm": arm, "seed": seed,
        "test_acc": float(result["test_acc"]),
        "test_ece": float(result["test_ece"]),
        "train_acc": float(result["train_acc"]),
        "gap": float(result["gap"]),
        "runtime_seconds": time.time() - started,
    }
    if "lam_final" in result:
        row["lam_final"] = float(result["lam_final"])
    atomic_json(out, {
        **expected, "status": "complete", "completed_utc": utc_now(),
        "row": row, "hist": result.get("hist", []),
    })
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default=",".join(TABULAR))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-new-cells", type=int, default=0)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    arms = args.arms.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]
    if any(d not in TABULAR for d in datasets):
        raise ValueError("non-tabular dataset in locked confirmatory run")
    if any(a not in ARMS for a in arms):
        raise ValueError("unknown arm")
    out_dir = Path(args.out).resolve()
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    cache_manifest = prepare_dataset_cache(out_dir)
    cache_manifest_sha = file_sha(cache_manifest)
    bundle = source_bundle()
    evidence_root = hashlib.sha256(
        f"{bundle['root_sha256']}|{cache_manifest_sha}".encode("utf-8")
    ).hexdigest().upper()
    atomic_json(out_dir / "source_bundle.json", {
        "generated_utc": utc_now(), **bundle,
        "dataset_cache_manifest_sha256": cache_manifest_sha,
        "evidence_root_sha256": evidence_root,
    })
    plan = [(d, a, s) for d in datasets for a in arms for s in seeds]
    valid_before = []
    todo = []
    for d, a, s in plan:
        path = out_dir / "runs" / f"{cell_key(d, a, s)}.json"
        expected = {"dataset": d, "arm": a, "seed": s,
                    "evidence_root_sha256": evidence_root}
        if path.exists() and valid_cell(path, expected):
            valid_before.append(str(path))
        else:
            todo.append((d, a, s, str(out_dir), str(cache_manifest.parent), evidence_root))
    if args.max_new_cells:
        todo = todo[:args.max_new_cells]
    print(f"PLAN total={len(plan)} valid={len(valid_before)} submitting={len(todo)}", flush=True)
    completed = list(valid_before)
    started = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_cell, cell) for cell in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                path = future.result()
                completed.append(path)
                atomic_json(out_dir / "progress.json", {
                    "utc": utc_now(), "planned_cells": len(plan),
                    "valid_before": len(valid_before),
                    "submitted_this_invocation": len(todo),
                    "completed_this_invocation": index,
                    "known_complete": len(completed),
                    "latest_cell": Path(path).name,
                    "elapsed_seconds": time.time() - started,
                    "evidence_root_sha256": evidence_root,
                })
                if index % 20 == 0 or index == len(todo):
                    print(f"PROGRESS {index}/{len(todo)} known={len(completed)}/{len(plan)} elapsed={time.time()-started:.1f}s", flush=True)

    all_paths = []
    for d, a, s in plan:
        path = out_dir / "runs" / f"{cell_key(d, a, s)}.json"
        expected = {"dataset": d, "arm": a, "seed": s,
                    "evidence_root_sha256": evidence_root}
        if path.exists() and valid_cell(path, expected):
            all_paths.append(path)
    if len(all_paths) != len(plan):
        print(f"PARTIAL valid={len(all_paths)}/{len(plan)}; rerun to resume", flush=True)
        return

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in all_paths]
    rows = [payload["row"] for payload in payloads]
    pd.DataFrame(rows).sort_values(["dataset", "arm", "seed"]).to_csv(
        out_dir / "confirmatory_raw.csv", index=False
    )
    atomic_json(out_dir / "summary.json", {
        "status": "complete", "generated_utc": utc_now(),
        "source_bundle": bundle,
        "dataset_cache_manifest_sha256": cache_manifest_sha,
        "evidence_root_sha256": evidence_root,
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "datasets": datasets, "arms": arms, "seeds": seeds,
        "cell_count": len(rows), "runtime_seconds_this_invocation": time.time() - started,
    })
    print(f"DONE cells={len(rows)} root={evidence_root}", flush=True)


if __name__ == "__main__":
    main()
