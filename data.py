"""Shared UCI HAR preparation with subject-disjoint validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
CHANNELS = [f"{sensor}_{axis}" for sensor in ["body_acc", "body_gyro", "total_acc"] for axis in "xyz"]

def load_raw(root, split):
    root = Path(root) / split
    x = np.stack([np.loadtxt(root / "Inertial Signals" / f"{c}_{split}.txt", dtype=np.float32, ndmin=2) for c in CHANNELS], axis=-1)
    y = np.loadtxt(root / f"y_{split}.txt", dtype=np.int64, ndmin=1) - 1
    subjects = np.loadtxt(root / f"subject_{split}.txt", dtype=np.int64, ndmin=1)
    if x.shape != (len(y), 128, 9) or len(subjects) != len(y):
        raise ValueError(f"Invalid {split} shapes: {x.shape}, {y.shape}, {subjects.shape}")
    if not np.isfinite(x).all() or not np.isin(y, np.arange(6)).all():
        raise ValueError("Signals must be finite and labels must be in 1..6")
    return x, y, subjects

def prepare(root, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y, subjects = load_raw(root, "train")
    xt, yt, st = load_raw(root, "test")
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(x, y, groups=subjects))
    groups = [set(subjects[train]), set(subjects[val]), set(st)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Subject leakage between train, validation, and test")
    mean = x[train].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(x[train].std(axis=(0, 1), dtype=np.float64), 1e-8).astype(np.float32)
    arrays = {"mean": mean, "std": std}
    for name, features, labels, ids in [("train", x[train], y[train], subjects[train]), ("val", x[val], y[val], subjects[val]), ("test", xt, yt, st)]:
        if set(labels) != set(range(6)):
            raise ValueError(f"{name} must contain all six activities")
        arrays.update({f"x_{name}": ((features - mean) / std).astype(np.float32), f"y_{name}": labels, f"subjects_{name}": ids})
    np.savez_compressed(output / "dataset.npz", **arrays)
    manifest = {"seed": seed, "channels": CHANNELS, "activities": ACTIVITIES, "normalization": "Per-channel z-score fitted on training subjects only", "splits": {name: {"windows": len(arrays[f"y_{name}"]), "subjects": sorted(map(int, np.unique(arrays[f"subjects_{name}"]))) } for name in ["train", "val", "test"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted UCI HAR Dataset folder")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.seed)
