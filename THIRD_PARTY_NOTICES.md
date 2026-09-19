# Third-party notices

The repository MIT license applies to original project code and original
experiment result artifacts. It does not replace dataset or dependency licenses.

## Benchmark datasets

The eight redistributed UCI-derived tabular datasets retain CC BY 4.0.
Their creators, source links, and transformations are listed in [DATA.md](DATA.md).
Phoneme, Fashion-MNIST images, and Electricity raw data are not redistributed.

## Constrained Parameter Regularization (CPR)

`src/tuning/vendor/pytorch_cpr/` contains the frozen upstream CPR implementation
used by the tuning experiment. It retains its Apache License 2.0, reproduced in
[`src/tuning/vendor/CPR_LICENSE`](src/tuning/vendor/CPR_LICENSE).
The vendored files are unmodified. The compatibility subclass in `src/tuning/core.py`
adapts a renamed PyTorch health-check method; it does not change CPR arithmetic.

## External dependencies

NumPy, pandas, SciPy, scikit-learn, PyTorch, torchvision, matplotlib, psutil, and
AdamP are external dependencies, not copied distributions. Obtain them from
their official projects/package indexes under their respective licenses.
