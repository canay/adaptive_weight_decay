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
