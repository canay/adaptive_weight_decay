# Artifact map

| Layer | Result data | Analysis/source | Role |
|---|---|---|---|
| Predecessor fixed grid and baselines | `results/predecessor/raw_runs.csv` | `src/predecessor/` | 870 runs; fixed-grid, direct baselines, and learning-rate transfer |
| Streaming | `results/predecessor/streaming/` | `src/predecessor/run_streaming_drift.py` | 120 delayed-label stream runs; separate design |
| Breadth archive | `results/predecessor/q1_suite/` | `src/predecessor/analyze_q1.py` | 1,128 archived rows; tabular grid/tuning references and a historical CNN subset |
| Locked tabular matrix | `results/locked/` | `src/locked/` | 810 runs, 18 task variants, nine predeclared arms, five seeds |
| CNN layer | `results/cnn/` | `src/cnn/` | 48 runs, two variants, eight effective arms, three seeds |
| Equal-update-budget factorial | `results/factorial/` | `src/factorial/`, `config/factorial.json` | 2,520 runs; separate corruption-interaction and donor-replay families |
| Method-specific search | `results/tuning/` | `src/tuning/`, `config/tuning.json` | 4,968 runs: 4,428 development trials and 540 selected refits |

`figures/` contains six empirical figure exports. These are not the manuscript
itself. `MD/02_design/ANALYSIS_AMENDMENT_20260826.md` preserves the historical
independence-safe analysis specification needed by the locked-matrix analyzer.

The zero-pass locked and tuning families do not establish equivalence. Rule B+
is retained as a predeclared inert variant in the locked records; it duplicates
rule B and must not be counted as independent supporting evidence.

## Export provenance

Per-run CSV metrics, scientific configurations, and frozen numerical summaries
are copied without changing their values. `PUBLIC_EXPORT.json` records the
single metadata redaction (a physical hostname in the selection record).
Internal historical hashes continue to name original artifacts, while the root
`SHA256SUMS.txt` names this public export.

One analysis-source repair is explicit: `src/predecessor/analyze_q1.py` filters
its paired comparisons to the 18 tabular variants, excluding the historical CNN
rows in the mixed archive. This restores agreement with the committed tabular
statistics. The raw rows and reference statistics were not changed.

Training sources are preserved for inspection, including their historical
private-workflow admission checks. They are not advertised as a portable,
one-command full-training release. The public entry point `reproduce.py`
reproduces the reported numerical analyses from the committed result tables.
