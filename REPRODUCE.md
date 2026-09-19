# Reproducing the reported analyses

## Analysis-only path

Use Python 3.12 and an isolated environment. No training, GPU, account, private
workspace, or network access is needed after installing the three dependencies.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements-analysis.txt
python reproduce.py --verify-only
python reproduce.py
```

The last command writes new outputs to the ignored `reproduced/` directory;
it does not overwrite the committed reference results. It:

- verifies public file checksums;
- reproduces the locked 810-run analysis and compares its six CSV summaries;
- recomputes the 48-run CNN means and sample standard deviations;
- recomputes the factorial's eight interaction tests and two feedback tests;
- recomputes all twelve tuning interactions, adjusted tests, bootstrap
  intervals, family verdict, and descriptive refit summaries;
- regenerates and compares predecessor fixed-grid and breadth statistics.

Failures stop the command. A passing run ends with
`ANALYSIS_REPRODUCTION_PASS` and writes `reproduced/validation.json`.
Comparisons use absolute tolerance 1e-12 and relative tolerance 1e-10.

The factorial and tuning wrappers load named pure statistical functions from
the supplied original analysis sources. Training-module imports are not needed
for this path. Seed-cell sensitivity results in the breadth archive are not
promoted to the locked-matrix claim gate.

## Training-source boundary

`src/` retains the historical training, analysis, controller, and optimizer
implementations. Some training entry points depend on the original directory
layout and hash-bound orchestration/admission records. Those private operational
records are intentionally excluded. Thus this snapshot supports executable
**result-level reproduction**, not a verified turnkey rerun of every training
job. Do not bypass the original admission checks or represent an adapted new
run as the historical run.

The exact observed factorial and tuning training environments are recorded in
`config/observed_training_environments.json`; they differ by platform, NumPy,
and PyTorch version. `requirements-analysis.txt` is the analysis environment,
not a claim that all experiments used one common environment. Bitwise equality
of new neural-network training is not promised across platforms.

For an input-level check, [DATA.md](DATA.md) provides the distributed arrays,
dataset attribution, optional Phoneme acquisition, and array-identity rules.
The CNN source uses the official Fashion-MNIST download, not a newly collected
dataset. Fresh full training requires a separately reviewed portable run plan.

## Empirical figures

The committed PNGs are the empirical exports used by the study. Original plotting
sources are retained under `src/predecessor/`. Regeneration requires matplotlib
in addition to the analysis dependencies; inspect each script's `--help` for its
input and figure-output directories. The layout-only study workflow diagram and
article sources are not part of this code/data repository.
