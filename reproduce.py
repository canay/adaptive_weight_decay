"""Reproduce committed statistics without training, downloads, or private files."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check_close(actual, expected, path="root"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"Key mismatch: {path}")
        for key in expected:
            check_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Length mismatch: {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            check_close(a, b, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not (math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
                or (math.isnan(actual) and math.isnan(expected))):
            raise AssertionError(f"Numerical mismatch: {path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"Value mismatch: {path}")


def scientific_functions(path, names):
    """Load named, unchanged pure functions, without importing training runtimes.

    The complete original source is shipped and bound by SHA256SUMS.txt.
    Only top-level function definitions named by this caller are executed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing scientific functions: {path.name}")
    namespace = {"np": np, "itertools": itertools}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def verify_checksums():
    manifest = ROOT / "SHA256SUMS.txt"
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT):
            raise AssertionError("Checksum path escapes repository")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest.lower():
            raise AssertionError(f"Checksum mismatch: {relative}")
        count += 1
    return count


def factorial():
    cfg = read_json(ROOT / "config/factorial.json")
    expected = read_json(ROOT / "results/factorial/decision.json")
    df = pd.read_csv(ROOT / "results/factorial/main_raw.csv")
    keys = ["dataset", "size", "noise", "arm", "control", "seed"]
    if len(df) != 2520 or df.duplicated(keys).any():
        raise AssertionError("Factorial matrix is incomplete or duplicated")
    index = df.set_index(keys)["test_acc"]
    ns = scientific_functions(ROOT / "src/factorial/analyze.py", ["contrast", "holm"])
    contrast, holm = ns["contrast"], ns["holm"]
    def score(d, z, n, a, c, s):
        return float(index.loc[(d, z, n, a, c, s)])
    primary = []
    for arm, baseline in itertools.product(["ctrlA", "ctrlB"], ["adadecay", "awd", "swd", "cwd"]):
        effects = [np.mean([
            score(d,z,.2,arm,"online",s)-score(d,z,.2,baseline,"online",s)
            -score(d,z,0.,arm,"online",s)+score(d,z,0.,baseline,"online",s)
            for z,s in itertools.product(cfg["sizes"],cfg["seeds"])]) for d in cfg["datasets"]]
        primary.append(dict(arm=arm, baseline=baseline, **contrast(effects, cfg)))
    feedback = []
    for arm in ["ctrlA", "ctrlB"]:
        effects = [np.mean([score(d,z,n,arm,"online",s)-score(d,z,n,arm,"replay",s)
                   for z,n,s in itertools.product(cfg["sizes"],cfg["noise"],cfg["seeds"])])
                   for d in cfg["datasets"]]
        feedback.append(dict(arm=arm, contrast="online-replay", **contrast(effects,cfg)))
    result = {"E1_primary": holm(primary,cfg), "E2_primary": holm(feedback,cfg)}
    for key in result:
        check_close(result[key], expected[key], key)
    return result


def tuning():
    cfg = read_json(ROOT / "config/tuning.json")
    expected = read_json(ROOT / "results/tuning/decision.json")
    df = pd.read_csv(ROOT / "results/tuning/all_trials.csv")
    keys = ["dataset", "noise", "phase", "arm", "hp_index", "seed"]
    if len(df) != 4968 or df.duplicated(keys).any():
        raise AssertionError("Tuning matrix is incomplete or duplicated")
    evaluation = df[df.phase == "evaluation"]
    if len(evaluation) != 540 or set(evaluation.status) != {"COMPLETE"}:
        raise AssertionError("Tuning evaluation matrix is not complete")
    metrics = ["accuracy", "ece", "nll", "train_accuracy", "train_nll", "clean_train_accuracy"]
    index = {(r.dataset, r.noise, r.arm, r.seed): {k: float(getattr(r,k)) for k in metrics}
             for r in evaluation.itertuples()}
    ns = scientific_functions(ROOT / "src/tuning/analyze.py", ["contrast", "holm", "family_summary", "calculate"])
    result = ns["calculate"](index,cfg)
    for key in result:
        check_close(result[key], expected[key], key)
    return result


def locked(output):
    ns = runpy.run_path(str(ROOT / "src/locked/analyze_confirmatory.py"), run_name="public_analysis")
    main = ns["main"]
    main.__globals__["RAW"] = ROOT / "results/locked/confirmatory_raw.csv"
    main.__globals__["OUT"] = output / "locked"
    main.__globals__["PROJECT"] = ROOT
    main()
    checked = []
    for reference in sorted((ROOT / "results/locked").glob("*.csv")):
        if reference.name == "confirmatory_raw.csv":
            continue
        reproduced = output / "locked" / reference.name
        pd.testing.assert_frame_equal(pd.read_csv(reproduced), pd.read_csv(reference),
                                      check_exact=False, rtol=1e-10, atol=1e-12)
        checked.append(reference.name)
    return checked


def cnn():
    with (ROOT / "results/cnn/architecture_raw.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_keys = set(itertools.product(["fashion_full","fashion_n20"],
                    ["none","fixed1","ctrlA","ctrlB","adadecay","awd","swd","cwd"], range(3)))
    keys = [(r["dataset"],r["arm"],int(r["seed"])) for r in rows]
    if len(rows) != 48 or set(keys) != expected_keys or len(set(keys)) != 48:
        raise AssertionError("CNN matrix is incomplete or duplicated")
    result = []
    for dataset, arm in sorted({(d,a) for d,a,_ in expected_keys}):
        group = sorted([r for r in rows if (r["dataset"],r["arm"]) == (dataset,arm)], key=lambda r:int(r["seed"]))
        accuracy = [float(r["test_acc"]) for r in group]
        result.append(dict(dataset=dataset,arm=arm,mean_test_acc=statistics.mean(accuracy),
            sd_test_acc=statistics.stdev(accuracy),mean_ece=statistics.mean(float(r["test_ece"]) for r in group),
            mean_gap=statistics.mean(float(r["gap"]) for r in group),seed_accuracies=accuracy))
    check_close(result, read_json(ROOT / "results/cnn/architecture_summary.json"), "cnn")
    return result


def predecessor(output):
    target = output / "predecessor"
    target.mkdir(parents=True, exist_ok=True)
    checks = []
    for subdir, raw, script, stats in [
        ("", "raw_runs.csv", "analyze_audit.py", "stats_audit.json"),
        ("q1_suite", "q1_raw.csv", "analyze_q1.py", "stats_q1.json"),
    ]:
        source = ROOT / "results/predecessor" / subdir
        dest = target / subdir
        dest.mkdir(exist_ok=True)
        shutil.copy2(source / raw, dest / raw)
        subprocess.run([sys.executable,str(ROOT / "src/predecessor" / script),"--dir",str(dest)],check=True)
        check_close(read_json(dest / stats), read_json(source / stats), "predecessor." + stats)
        checks.append(stats)
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--output", type=Path, default=ROOT / "reproduced")
    args = ap.parse_args()
    count = verify_checksums()
    if args.verify_only:
        print(f"CHECKSUMS_PASS {count} files")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"factorial":factorial(), "tuning":tuning(), "cnn":cnn()}
    results["locked_csv_files"] = locked(args.output)
    results["predecessor_statistics"] = predecessor(args.output)
    results["checksums_verified"] = count
    results["status"] = "PASS"
    results["scope"] = "Analysis-only numerical reproduction; no training or network access"
    (args.output / "validation.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print("ANALYSIS_REPRODUCTION_PASS: locked=810; cnn=48; factorial=2520; tuning=4968")


if __name__ == "__main__":
    main()
