from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


EXPECTED_DATASETS = {"fashion_full", "fashion_n20"}
EXPECTED_ARMS = {"none", "fixed1", "ctrlA", "ctrlB", "adadecay", "awd", "swd", "cwd"}
EXPECTED_SEEDS = {0, 1, 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_architecture.py <outputs/main> <analysis-dir>")
    out = Path(sys.argv[1]).resolve()
    analysis = Path(sys.argv[2]).resolve()
    analysis.mkdir(parents=True, exist_ok=True)

    rows = []
    keys = set()
    for path in sorted((out / "runs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete cell: {path}")
        row = payload["row"]
        key = (row["dataset"], row["arm"], int(row["seed"]))
        if key in keys:
            raise RuntimeError(f"duplicate cell: {key}")
        keys.add(key)
        rows.append(row)
    expected = {(d, a, s) for d in EXPECTED_DATASETS for a in EXPECTED_ARMS for s in EXPECTED_SEEDS}
    if keys != expected:
        raise RuntimeError(f"matrix mismatch missing={sorted(expected-keys)} extra={sorted(keys-expected)}")

    raw_path = analysis / "architecture_raw.csv"
    fields = ["dataset", "arm", "seed", "test_acc", "test_ece", "train_acc", "gap", "runtime_seconds"]
    with raw_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])

    summary = []
    for dataset in sorted(EXPECTED_DATASETS):
        for arm in sorted(EXPECTED_ARMS):
            group = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            summary.append({
                "dataset": dataset,
                "arm": arm,
                "mean_test_acc": statistics.mean(r["test_acc"] for r in group),
                "sd_test_acc": statistics.stdev(r["test_acc"] for r in group),
                "mean_ece": statistics.mean(r["test_ece"] for r in group),
                "mean_gap": statistics.mean(r["gap"] for r in group),
                "seed_accuracies": [r["test_acc"] for r in sorted(group, key=lambda x: x["seed"])],
            })
    summary_path = analysis / "architecture_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fashion-MNIST architecture-layer report",
        "",
        "Date/time: 2026-08-26 14:36 +03:00  ",
        "Tool: Codex  ",
        "Model, if known: GPT-5.6 Extra High (xhigh, user-attested; runtime exposed GPT-5 family)  ",
        "Operation ID: `F06-NOVELTY-ACCEPTABILITY-UPGRADE-20260826-01`",
        "",
        "Verdict: `COMPLETE / DESCRIPTIVE_ARCHITECTURE_CHECK`.",
        "",
        "The layer contains 48/48 unique cells (two variants, eight arms, three matched seeds). "
        "With only two variants and three seeds, it is a descriptive architecture check; no inferential superiority claim is made.",
        "",
        "| Variant | Arm | Mean accuracy | SD | Seed accuracies | Mean ECE |",
        "|---|---|---:|---:|---|---:|",
    ]
    for dataset in sorted(EXPECTED_DATASETS):
        group = sorted((x for x in summary if x["dataset"] == dataset), key=lambda x: -x["mean_test_acc"])
        for x in group:
            vals = ", ".join(f"{v:.4f}" for v in x["seed_accuracies"])
            lines.append(f"| {dataset} | {x['arm']} | {x['mean_test_acc']:.4f} | {x['sd_test_acc']:.4f} | {vals} | {x['mean_ece']:.4f} |")
    lines.extend([
        "",
        "## Boundary interpretation",
        "",
        "The ranking is reported separately for clean and corrupted data. A cross-regime reversal may corroborate the tabular boundary finding, "
        "but this small CNN layer cannot establish architecture-general behavior. The raw rows, all seed values, and data/source hashes remain the claim authority.",
        "",
        f"Raw CSV SHA-256: `{sha256(raw_path)}`.  ",
        f"Summary JSON SHA-256: `{sha256(summary_path)}`.",
        "",
    ])
    (analysis / "ARCHITECTURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
