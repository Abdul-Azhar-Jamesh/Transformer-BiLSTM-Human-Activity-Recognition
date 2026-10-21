# Human Activity Recognition: BiLSTM and Transformer

A focused comparison of a stacked bidirectional LSTM and a supervised patch Transformer on UCI HAR. Includes data preparation, training, checkpoint selection, and test evaluation. Both models use identical subject splits, normalization, nine sensor channels, and activity labels.

## Models

| Model | Architecture | Framework |
| --- | --- | --- |
| BiLSTM | Bidirectional(LSTM(128)) -> Bidirectional(LSTM(64)) -> Dense(128) -> Dense(64) -> 6 classes | TensorFlow/Keras |
| Transformer | 16-step patches -> 128-dimensional embeddings -> 4 encoder blocks, 4 heads -> average pooling -> 6 classes | PyTorch |

See [SOURCES.md](SOURCES.md) for component credits and licensing. The BiLSTM uses two bidirectional recurrent layers, with no convolution or attention. The Transformer is trained with labels from scratch.

## Setup

Use Python 3.11 or newer with versions supported by both frameworks. Commands below are for PowerShell from this folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the exact direct dependency versions used during local validation, see `requirements-tested.txt`.

On macOS/Linux activate with `source .venv/bin/activate`. PyTorch automatically uses CUDA when available; native Windows TensorFlow uses CPU with current pip packages. Use WSL2 and an appropriate TensorFlow GPU installation for TensorFlow GPU training. `--device` controls PyTorch only.

## Prepare data

Obtain and extract UCI HAR (Human Activity Recognition Using Smartphones). Point `--dataset` at the folder containing `train` and `test`, each with `Inertial Signals`. No dataset is bundled.

```powershell
python data.py --dataset "C:\path\to\UCI HAR Dataset"
```

Windows are already 128 samples long at 50 Hz; no additional windowing is applied. The channel order is body acceleration XYZ, body gyroscope XYZ, total acceleration XYZ. Official labels become 0..5: walking, walking upstairs, walking downstairs, sitting, standing, laying.

The official test partition is retained. A seeded group split reserves 20% of the actual official training subjects for validation; windows from one person cannot cross partitions. Per-channel normalization is fitted on the remaining training windows. `data/processed/manifest.json` records subject IDs, counts, channels, and labels; `dataset.npz` includes normalization parameters. Do not change the prepared data between model runs.

## Train

```powershell
python train.py --model bilstm --output runs/bilstm
python train.py --model transformer --output runs/transformer
```

Both default to 100 epochs, batch size 64, Adam at 0.001, seed 42, and early stopping after 15 epochs without improvement in validation macro F1. The best checkpoint is saved by validation macro F1, along with `config.json` and `history.csv`. Existing nonempty output directories are protected; choose a new `--output` for another experiment. Seeds improve repeatability but do not guarantee identical results across frameworks or hardware.

For a short pipeline check use `--epochs 1`; this is not a meaningful benchmark. Train both to convergence before comparing performance. The frameworks and model sizes differ, so identical settings do not establish equal compute budgets. TensorFlow includes dense-layer L2 regularization.

## Evaluate and compare

```powershell
python evaluate.py --run runs/bilstm
python evaluate.py --run runs/transformer
python compare.py runs/bilstm runs/transformer
```

Evaluation loads the best validation checkpoint and uses the official held-out test data. Each `runs/<model>/evaluation` contains accuracy, macro F1, a per-class classification report, confusion matrix CSV/PNG, and predictions with probabilities. Comparison writes `runs/comparison.csv`. A dataset SHA-256 check prevents accidental evaluation or comparison against different prepared data. Use the test set for final reporting; repeated test-driven tuning compromises its held-out role.

No benchmark accuracy is claimed in this repository. Scores must come from your own saved evaluations. Full training can take substantial time on CPU.

## Hardware reference

The [hardware setup guide](docs/HARDWARE_SETUP_GUIDE.md) includes ESP32 / Arduino Uno/Nano + MPU-6050 components, pin connections, calibration, firmware upload, serial/BLE formats, and the channel mapping required by these models. Firmware sketches are included in `hardware/firmware/`.

The boards stream sensor data to a PC. Live sensor inference is not wired into the training/evaluation commands; the guide explains the preprocessing needed for that integration.

## Checks

```powershell
python -m unittest discover -s tests -v
```

See [VALIDATION.md](VALIDATION.md) for project validation details.

## GitHub

Raw data, processed arrays, environments, weights, and generated runs are excluded by `.gitignore`. Component credits and licensing are documented in [SOURCES.md](SOURCES.md).

A local repository is already initialized on `main`. To record the source files:

```powershell
git add .
git commit -m "Add BiLSTM and Transformer HAR comparison"
```

Create an empty repository on GitHub, then add its remote and push. No GitHub repository is created or published by these files.
