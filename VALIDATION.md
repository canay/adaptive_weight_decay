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
