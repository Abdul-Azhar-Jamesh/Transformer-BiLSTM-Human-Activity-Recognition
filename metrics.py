"""Consistent metrics and figures for either framework."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data import ACTIVITIES

def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=list(range(6)), average="macro", zero_division=0))}

def write_report(y, probabilities, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = probabilities.argmax(axis=1)
    report = scores(y, pred)
    report["classification_report"] = classification_report(y, pred, labels=list(range(6)), target_names=ACTIVITIES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, pred, labels=list(range(6)))
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    np.savez_compressed(output / "predictions.npz", labels=y, predictions=pred, probabilities=probabilities)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=ACTIVITIES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["accuracy", "macro_f1"]}, indent=2))
    return report
