# Selection-Sensitive Regime Effects in Validation-Free AdamW Weight-Decay Control

Research code and reproducibility artifacts for the study by **Ozkan Canay**.
This repository compares validation-free AdamW weight-decay controllers across
data regimes and configuration-selection policies. It does not claim general
controller superiority.

## Start here

1. Read [REPRODUCE.md](REPRODUCE.md) for the analysis-only reproduction commands.
2. Read [ARTIFACTS.md](ARTIFACTS.md) for the experiment layers and file mapping.
3. Read [DATA.md](DATA.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
   before using the benchmark inputs.

The public snapshot includes the predecessor experiments, the 810-run locked
tabular matrix, the 48-run convolutional layer, the 2,520-run equal-update-budget
factorial, and the 4,968-run method-specific-search experiment. These are
distinct designs; their rows and test families must not be pooled.

Analysis can be reproduced from the committed per-run result tables without
training models or downloading benchmark data. Historical training sources and
frozen configurations are also supplied. See the reproduction guide for the
difference between result reproduction and a fresh training run.

## Licensing and citation

Original project code and original result artifacts are covered by the
[MIT License](LICENSE). Benchmark datasets and vendored third-party code retain
their own licenses; the repository license does not replace them.

Citation metadata is provided in [CITATION.cff](CITATION.cff). This is a research
artifact snapshot, not a statement of journal acceptance. No article DOI is
asserted. The manuscript and private editorial records are not included.
