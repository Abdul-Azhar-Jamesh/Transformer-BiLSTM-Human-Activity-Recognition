"""Integration checks for data separation and model output contracts."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from data import CHANNELS, prepare

class DataTests(unittest.TestCase):
    def test_subject_split_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(7)
            for split, people, offset in [("train", [1, 3, 5, 7, 9], 0), ("test", [2, 4], 100)]:
                subjects = np.repeat(people, 6)
                folder = root / split
                (folder / "Inertial Signals").mkdir(parents=True)
                np.savetxt(folder / f"subject_{split}.txt", subjects, fmt="%d")
                np.savetxt(folder / f"y_{split}.txt", np.tile(np.arange(1, 7), len(people)), fmt="%d")
                for channel in CHANNELS:
                    np.savetxt(folder / "Inertial Signals" / f"{channel}_{split}.txt", rng.normal(offset, 1, (len(subjects), 128)))
            prepare(root, root / "processed")
            with np.load(root / "processed/dataset.npz") as data:
                sets = [set(data[f"subjects_{s}"]) for s in ["train", "val", "test"]]
                self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                np.testing.assert_allclose(data["x_train"].mean(axis=(0, 1)), 0, atol=1e-6)
                np.testing.assert_allclose(data["x_train"].std(axis=(0, 1)), 1, atol=1e-5)
                self.assertGreater(float(data["x_test"].mean()), 90)
                self.assertEqual(data["x_train"].shape[1:], (128, 9))

class ModelTests(unittest.TestCase):
    def test_transformer_output_and_gradient(self):
        import torch
        from models.supervised_transformer import SupervisedTransformer
        torch.set_num_threads(2)
        model = SupervisedTransformer(9, 16, 128, 4, 4, 8, 6)
        logits = model(torch.randn(2, 128, 9))
        self.assertEqual(tuple(logits.shape), (2, 6))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_bilstm_output(self):
        import tensorflow as tf
        from models.bilstm import build_bilstm
        model = build_bilstm()
        self.assertEqual(sum(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers), 2)
        probabilities = model(tf.zeros((2, 128, 9)), training=False).numpy()
        self.assertEqual(probabilities.shape, (2, 6))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
