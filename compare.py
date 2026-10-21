"""Compare evaluation metrics from compatible experiments."""
import argparse
import csv
import json
from pathlib import Path

def compare(runs, output):
    rows, hashes = [], set()
    for run in map(Path, runs):
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "evaluation/metrics.json").read_text())
        hashes.add(config["dataset_sha256"])
        rows.append(dict(model=config["model"], run=str(run), accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
    if len(hashes) != 1:
        raise ValueError("Runs used different prepared data; comparison would be invalid")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']:12s} accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="runs/comparison.csv")
    args = parser.parse_args()
    compare(args.runs, args.output)
