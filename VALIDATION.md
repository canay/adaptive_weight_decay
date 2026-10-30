# Public snapshot validation

Date: 2026-09-19. Scope: numerical result reproduction, not new training.

The public analysis entry point completed successfully on Python 3.12.12 with
NumPy 2.5.2, pandas 2.3.3, and SciPy 1.18.0. It reproduced the locked matrix's
six CSV summaries, the CNN summaries, factorial E1/E2 statistics, tuning E4
statistics and descriptive summaries, and predecessor fixed-grid/breadth
statistics within the documented tolerances.

The optional Phoneme downloader was also tested: all six processed arrays
matched the historical shape, dtype, and array SHA-256. The downloaded data
was kept outside this public repository and is not redistributed.

The source fix that makes the historical breadth analyzer use only its
18 tabular variants is documented in `PUBLIC_EXPORT.json` and `ARTIFACTS.md`.
No per-run metric or committed reference statistic was changed.

This check does not certify fresh-training equivalence, journal acceptance,
or a DOI deposit. The repository checksum manifest binds the exported bytes.

## Figure refresh (2026-09-29)

The six PNGs in `figures/` and three plotting sources in `src/predecessor/`
(`make_figures_audit.py`, `make_q1_figures.py`, `make_streaming_figure.py`) were
updated to the versions behind the study's current figures. The changes are
presentational: print-width layout, type sizes, series names, and markers,
line styles and hatching that stay distinguishable in grayscale. No result
table, statistic, or analysis source changed. Apart from these files,
`REPRODUCE.md` (which now lists the regeneration commands) and this note,
every file in the checksum manifest is byte-identical to the 2026-09-19
snapshot.

Run on the committed result tables alone, the three plotting sources
reproduced all six committed PNGs byte for byte (Python 3.12.7, matplotlib
3.11.1, Pillow 12.3.0, NumPy 2.3.5, pandas 2.3.3). On the same environment
`python reproduce.py` ended with `ANALYSIS_REPRODUCTION_PASS`, and the checksum
manifest was regenerated for the changed files.

## Figure refresh (2026-10-04)

The six PNGs in `figures/` and the three plotting sources in `src/predecessor/`
were updated again to the versions behind the study's current figures. The
changes are presentational: smaller type sizes with the 7 pt print floor kept,
and thinner data lines in the sensitivity, trajectory and signal-gap figures.
No result table, statistic, configuration, or analysis source changed. Apart
from these files and this note, every file in the checksum manifest is
byte-identical to the 2026-09-29 snapshot.

Run on the committed result tables alone, the three plotting sources
reproduced all six committed PNGs byte for byte (Python 3.12.12, matplotlib
3.11.1, Pillow 12.1.0, NumPy 2.5.2, pandas 2.3.3). The checksum
manifest was regenerated for the changed files and `python reproduce.py
--verify-only` passed.

## Figure 7 correction (2026-10-17)

`figures/fig7_pareto.png` and its plotting source `src/predecessor/make_q1_figures.py` were
updated to the versions behind the study's current Figure 7. The oracle reference point is now
plotted at seven training runs, the best of the seven fixed-grid runs, instead of eight; the
validation-tuning point stays at eight runs, and the x-axis ticks read 1, 7 and 8. No accuracy
value, result table, statistic, configuration, or analysis source changed. Apart from these two
files and this note, every file in the checksum manifest is byte-identical to the 2026-10-04
snapshot.

Run on the committed result tables alone, `src/predecessor/make_q1_figures.py` reproduced the
committed `fig7_pareto.png` and `fig8_regime.png` byte for byte (Python 3.12.12,
matplotlib 3.11.1, Pillow 12.0.0, NumPy 2.3.5, pandas 2.3.3).
The checksum manifest was regenerated for the changed files and `python reproduce.py
--verify-only` passed.

## Figure type-size refresh (2026-10-29)

The five PNGs `fig1_sensitivity.png`, `fig2_lambda_traj.png`, `fig4_signal_gap.png`,
`fig6_streaming.png` and `fig8_regime.png` in `figures/`, and the three plotting sources in
`src/predecessor/`, were updated to the versions behind the figures of the single-column review
copy requested by the journal. The changes are presentational: one in-figure text size per figure
at the 7 pt print floor, a shared y label per row where repeated labels collided, and the matching
font settings. No data series, panel, result table, statistic, configuration, or analysis source
changed, and `fig7_pareto.png` is unchanged. Apart from these files and this note, every file in the
checksum manifest is byte-identical to the 2026-10-17 snapshot.

Run on the committed result tables alone, the three plotting sources reproduced all six committed
PNGs byte for byte (Python 3.12.12, matplotlib 3.11.1, Pillow 12.1.0, NumPy 2.5.2, pandas 2.3.3).
The checksum manifest was regenerated for the changed files and `python reproduce.py
--verify-only` passed.

## Run identifier and provenance cleanup (2026-10-30)

Thirteen files were edited to remove the names of workflow tools from run identifiers and
provenance headers. Four run identifiers now use neutral names taken from the study's own layer
labels: `2026-08-26_strong_baseline_fidelity`, `2026-08-26_locked_confirmatory_matrix`,
`2026-09-04_equal_update_budget_factorial` and `2026-09-04_method_specific_search`. They replace
the `run_id` value in `config/factorial.json`, `config/tuning.json` and
`results/tuning/decision.json`, and the predecessor run folder named in `src/cnn/run_fashion.py`,
`src/locked/run_confirmatory.py` and `src/factorial/prepare.py`. Tool and model provenance lines
were removed from `MD/02_design/ANALYSIS_AMENDMENT_20260826.md`, from the report headers written by
`src/cnn/analyze_architecture.py` and `src/locked/analyze_confirmatory.py`, and from the docstrings
of `src/tuning/core.py`, `src/tuning/data.py` and `src/tuning/engine.py`; the `tool` and `model`
fields were removed from `data/INPUT_LINEAGE.json`. No number, result table, statistic,
configuration value, or analysis step changed. `PUBLIC_EXPORT.json` lists the original and public
SHA-256 of each edited file.

`results/locked/analysis_summary.json` records the SHA-256 of the analysis amendment and of
`src/locked/analyze_confirmatory.py` as they were when the locked analysis ran. Those values now
describe the pre-cleanup bytes, which remain in this repository's history at commit `195f776`. The
training sources name their predecessor run folders by the neutral identifiers, which differ from
the private folder names; as before, these sources are supplied for inspection, not as a
one-command training release. Apart from these files, `ARTIFACTS.md`, `PUBLIC_EXPORT.json` and this
note, every file in the checksum manifest is byte-identical to the 2026-10-29 snapshot.

On the committed result tables, `python reproduce.py` ended with `ANALYSIS_REPRODUCTION_PASS`
(Python 3.12.12, NumPy 2.5.2, pandas 2.3.3, SciPy 1.18.0). The checksum manifest was regenerated
for the changed files and `python reproduce.py --verify-only` passed.
