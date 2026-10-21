"""Evaluate the best validation checkpoint on the untouched official test split."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import fingerprint
from metrics import write_report

def evaluate(args):
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    if config["model"] not in {"bilstm", "transformer"}:
        raise ValueError("This run uses a replaced model. Train a new BiLSTM or Transformer run.")
    dataset = Path(args.data) / "dataset.npz"
    if fingerprint(dataset) != config["dataset_sha256"]:
        raise ValueError("Prepared dataset differs from training; use the original dataset.npz")
    with np.load(dataset) as archive:
        x, y = archive["x_test"], archive["y_test"]
    if config["model"] == "bilstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(run / "best.keras", compile=False)
        probabilities = model.predict(x, batch_size=args.batch_size, verbose=0)
    else:
        import torch
        from models.supervised_transformer import SupervisedTransformer
        torch.set_num_threads(4)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = SupervisedTransformer(**config["transformer"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            probabilities = np.concatenate([model(torch.from_numpy(x[i:i + args.batch_size]).to(device)).softmax(dim=-1).cpu().numpy() for i in range(0, len(x), args.batch_size)])
    write_report(y, probabilities, run / "evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    evaluate(parser.parse_args())
