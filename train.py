"""Train a BiLSTM or supervised Transformer on the same prepared data."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
from data import fingerprint
from metrics import scores

TRANSFORMER = dict(in_channels=9, patch_size=16, embed_dim=128, num_heads=4, num_layers=4, max_len=8, num_classes=6, dropout_p=0.1)

def train(args):
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, patience, and learning rate must be positive")
    output = Path(args.output or f"runs/{args.model}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new --output; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = Path(args.data) / "dataset.npz"
    with np.load(dataset) as archive:
        x, y, xv, yv = [archive[k] for k in ["x_train", "y_train", "x_val", "y_val"]]
    config = vars(args).copy()
    config.update(dataset_sha256=fingerprint(dataset), transformer=TRANSFORMER, input_shape=[128, 9])
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if args.model == "bilstm":
        import tensorflow as tf
        from models.bilstm import build_bilstm
        tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()
        model = build_bilstm()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        class ValidationF1(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                metrics = scores(yv, self.model.predict(xv, batch_size=args.batch_size, verbose=0).argmax(axis=1))
                logs["val_macro_f1"] = metrics["macro_f1"]
        model.fit(x, y, validation_data=(xv, yv), batch_size=args.batch_size, epochs=args.epochs, callbacks=[ValidationF1(), tf.keras.callbacks.ModelCheckpoint(str(output / "best.keras"), monitor="val_macro_f1", mode="max", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.patience), tf.keras.callbacks.CSVLogger(str(output / "history.csv"))], verbose=2)
    else:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models.supervised_transformer import SupervisedTransformer
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed_all(args.seed)
        model = SupervisedTransformer(**TRANSFORMER).to(device)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(TensorDataset(torch.from_numpy(xv), torch.from_numpy(yv)), batch_size=args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        best, stale = -1.0, 0
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "val_loss", "val_accuracy", "val_macro_f1"])
            writer.writeheader()
            for epoch in range(args.epochs):
                model.train()
                loss_sum = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(yb)
                model.eval()
                predictions, val_loss = [], 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits = model(xb.to(device))
                        val_loss += criterion(logits, yb.to(device)).item() * len(yb)
                        predictions.extend(logits.argmax(dim=1).cpu().tolist())
                metrics = scores(yv, predictions)
                row = dict(epoch=epoch + 1, loss=loss_sum / len(y), val_loss=val_loss / len(yv), val_accuracy=metrics["accuracy"], val_macro_f1=metrics["macro_f1"])
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
                if metrics["macro_f1"] > best:
                    best, stale = metrics["macro_f1"], 0
                    torch.save(model.state_dict(), output / "best.pt")
                else:
                    stale += 1
                if stale >= args.patience:
                    break
    print(f"Best validation checkpoint saved in {output}. Run evaluate.py for test metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["bilstm", "transformer"])
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--threads", type=int, default=4)
    train(parser.parse_args())
