# Benchmark inputs and attribution

The committed NPZ files are processed benchmark inputs, not newly collected
participant data. They retain the source dataset licenses below. **They are
not relicensed under the repository's MIT license.** License records were
checked on 2026-09-19.

## CC BY 4.0 inputs

These source records specify [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain the attribution, source link, license link, and transformation notice
when redistributing the processed arrays. No source creator endorses this study.

| Input | Source credit and dataset record | Acquisition identity |
|---|---|---|
| Adult | B. Becker and R. Kohavi, [Adult](https://doi.org/10.24432/C5XW20) | OpenML 1590 |
| Bank | S. Moro, P. Rita, and P. Cortez, [Bank Marketing](https://doi.org/10.24432/C5K306) | OpenML 1461 |
| Digits | E. Alpaydin and C. Kaynak, [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49) | scikit-learn `load_digits`, UCI test-set copy |
| Spambase | M. Hopkins, E. Reeber, G. Forman, and J. Suermondt, [Spambase](https://doi.org/10.24432/C53G6X) | OpenML 44 |
| QSAR | K. Mansouri, T. Ringsted, D. Ballabio, R. Todeschini, and V. Consonni, [QSAR biodegradation](https://doi.org/10.24432/C5H60M) | OpenML 1494 |
| Waveform | L. Breiman and C. J. Stone, [Waveform Database Generator v2](https://doi.org/10.24432/C56014) | OpenML 60, waveform-5000 |
| MFeat | R. Duin, [Multiple Features](https://doi.org/10.24432/C5HC70) | OpenML 16, Karhunen subset |
| Satimage | A. Srinivasan, [Statlog Landsat Satellite](https://doi.org/10.24432/C55887) | OpenML 182 |

The upstream landing pages are the UCI Machine Learning Repository records
identified by these dataset DOIs. The transformation code is provided in
`code/common6.py`, `code/q1_suite.py`, and `src/locked/run_confirmatory.py`.

## Transformations and identities

- `data/adult.npz` and `data/bank_marketing.npz` are the historical encoded
  10,500/4,500 train/test caches. Categorical encoding and train-fitted scaling
  precede the task-specific transformations in `common6.py`.
- `data/digits.npz` contains the historical image feature/label cache. The
  loader divides pixel values by 16, applies a seed-0 70/30 split, and fits
  normalization on training data.
- `data/factorial/*_full.npz` contains the frozen full clean task arrays.
  Subsampling, synthetic training-label corruption, and train-only safe scaling
  are applied by the factorial runner. These are derived forms of the cited
  datasets, not the unmodified upstream downloads.
- `data/tuning/` separates original training and test arrays. The committed
  `*_partition.json` files record the class-stratified development/validation
  indices and noise seeds. `INPUT_LINEAGE.json` identifies every historical
  input. Validation selection never opens original test labels.
- `ARRAY_IDENTITIES.json` supplies array shape, dtype, and raw-array SHA-256.
  `SHA256SUMS.txt` binds the bytes actually distributed in this repository.
  ZIP compression/version differences may change an NPZ file hash without
  changing any of its arrays; these two identities are intentionally distinct.

## Inputs not redistributed

**Phoneme:** [OpenML 1489](https://www.openml.org/d/1489) exposes the license
field as `Public`, without a sufficiently specific formal redistribution
license in the checked record. This repository therefore omits its raw and
processed feature/label arrays. It retains result metrics, partition indices,
and array hashes. Review the upstream terms before optional acquisition:

```bash
python -m pip install scikit-learn==1.9.0
python prepare_phoneme.py --download
```

The script fails on any array-identity drift. It writes only to the ignored
`reproduced/phoneme/` directory by default; do not add these downloads to Git.

**Fashion-MNIST:** obtain the dataset from the
[official Zalando Research repository](https://github.com/zalandoresearch/fashion-mnist),
which provides download hashes and its MIT license. Its raw image archive is
not duplicated here. The committed CNN per-run results do not require this
download for analysis reproduction.

**Streaming:** the Electricity benchmark is not redistributed. The supplied
`src/predecessor/run_streaming_drift.py` records its OpenML acquisition and
the seeded rotating-hyperplane/SEA generators. The 120-run streaming metrics
and trajectories are supplied separately from the tabular experiments.
