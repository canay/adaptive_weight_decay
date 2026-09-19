"""Optional upstream acquisition; Phoneme raw data is not redistributed here."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduce import ROOT, scientific_functions, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True,
                        help="Explicitly download OpenML 1489; review its source terms first")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced/phoneme")
    args = parser.parse_args()
    from sklearn.datasets import fetch_openml
    raw = fetch_openml(data_id=1489, as_frame=False, data_home=str(args.output / "openml-cache"))
    x = np.asarray(raw.data, dtype=np.float64)
    classes = sorted(set(np.asarray(raw.target).tolist()))
    mapping = {value:index for index,value in enumerate(classes)}
    y = np.array([mapping[value] for value in np.asarray(raw.target).tolist()], dtype=np.int64)
    fn = scientific_functions(ROOT / "src/locked/run_confirmatory.py", ["_openml_variant"])["_openml_variant"]
    arrays = fn(x,y,False)
    expected = read_json(ROOT / "data/ARRAY_IDENTITIES.json")["phoneme_full"]["arrays"]
    assert set(arrays) == set(expected)
    for key, array in arrays.items():
        actual = {"shape":list(array.shape), "dtype":str(array.dtype),
                  "sha256":hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
        if actual != expected[key]:
            raise RuntimeError("UPSTREAM_OR_PREPROCESSING_DRIFT: " + key)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "phoneme_full.npz", **arrays)
    for suffix, xkey, ykey in [("train","Xtr","ytr"),("test","Xte","yte")]:
        np.savez_compressed(args.output / f"phoneme_{suffix}.npz", X=arrays[xkey],y=arrays[ykey],n_class=int(arrays["n_class"]))
    print("PHONEME_ARRAY_IDENTITIES_PASS: all six arrays match the historical inputs")
    print("Data remains local; this script grants no redistribution rights.")


if __name__ == "__main__":
    main()
