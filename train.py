"""
train.py
========
Training loop for the Optical Neural Network on MNIST.

Optimization strategy: Finite Difference Gradient Estimation
--------------------------------------------------------------
Standard backpropagation requires differentiable operations throughout.
Physical ONNs on hardware can't backpropagate — you can only measure
inputs and outputs (forward passes only).

We use finite differences:
    grad ≈ (loss(φ + ε) - loss(φ - ε)) / (2ε)

This is called "zeroth-order" or "black-box" optimization and is
directly applicable to real hardware ONNs — a key point to mention
in your KAUST application!

For simulation speed, we use a mini-batch stochastic version:
estimate gradients on small batches rather than one sample at a time.
"""

import numpy as np
import time
from tqdm import tqdm

from data.mnist_loader import load_mnist
from onn.network import ONN
from onn.utils import softmax, cross_entropy_loss, accuracy


# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    "input_size":   64,       # ONN input dimension (8×8 downsampled MNIST)
    "num_layers":   3,        # number of optical layers
    "layer_depth":  2,        # MZI columns per layer
    "num_classes":  10,       # digits 0–9
    "epochs":       15,       # training epochs
    "lr":           0.05,     # learning rate for phase updates
    "epsilon":      1e-3,     # finite difference step size
    "batch_size":   32,       # samples per gradient estimate
    "n_train":      3000,     # training samples
    "n_test":       500,      # test samples
    "save_path":    "onn_trained_phases.npy",
}
# ──────────────────────────────────────────────────────────────────────────────


def batch_loss(model: ONN, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over a batch.

    Args:
        model:   ONN model
        X_batch: input batch, shape (B, input_size)
        y_batch: label batch, shape (B,)

    Returns:
        mean loss (scalar)
    """
    total = 0.0
    for x, label in zip(X_batch, y_batch):
        scores = model.forward(x)
        probs  = softmax(scores)
        total += cross_entropy_loss(probs, label)
    return total / len(X_batch)


def finite_diff_step(model: ONN,
                     X_batch: np.ndarray,
                     y_batch: np.ndarray,
                     lr: float,
                     epsilon: float):
    """
    One finite-difference gradient update over all phase parameters.

    For each phase φ in the network:
        grad = (L(φ+ε) - L(φ-ε)) / (2ε)
        φ ← φ - lr * grad

    Args:
        model:   ONN model (modified in place)
        X_batch: mini-batch inputs
        y_batch: mini-batch labels
        lr:      learning rate
        epsilon: finite difference step
    """
    for layer in model.layers:
        for col in range(layer.phases.shape[0]):
            for mzi_i in range(layer.phases.shape[1]):
                for phi_j in range(2):                    # phi1 and phi2
                    original = layer.phases[col, mzi_i, phi_j]

                    # Forward nudge
                    layer.phases[col, mzi_i, phi_j] = original + epsilon
                    loss_plus = batch_loss(model, X_batch, y_batch)

                    # Backward nudge
                    layer.phases[col, mzi_i, phi_j] = original - epsilon
                    loss_minus = batch_loss(model, X_batch, y_batch)

                    # Gradient estimate & update
                    grad = (loss_plus - loss_minus) / (2 * epsilon)
                    layer.phases[col, mzi_i, phi_j] = original - lr * grad


def train():
    print("=" * 60)
    print("  Optical Neural Network — MNIST Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_mnist(
        n_train=CONFIG["n_train"],
        n_test=CONFIG["n_test"],
        input_size=CONFIG["input_size"]
    )

    # ── Build model ───────────────────────────────────────────────
    model = ONN(
        input_size=CONFIG["input_size"],
        num_layers=CONFIG["num_layers"],
        layer_depth=CONFIG["layer_depth"],
        num_classes=CONFIG["num_classes"]
    )
    print(f"\nModel built: {CONFIG['num_layers']} optical layers × "
          f"depth {CONFIG['layer_depth']}")
    print(f"Trainable phase parameters: {model.count_parameters()}")

    # ── Training loop ─────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    t0 = time.time()

    for epoch in range(CONFIG["epochs"]):
        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches  = len(X_train) // CONFIG["batch_size"]

        for b in tqdm(range(n_batches),
                      desc=f"Epoch {epoch+1:02d}/{CONFIG['epochs']}",
                      ncols=70):
            start = b * CONFIG["batch_size"]
            end   = start + CONFIG["batch_size"]
            X_b   = X_shuf[start:end]
            y_b   = y_shuf[start:end]

            finite_diff_step(model, X_b, y_b,
                             lr=CONFIG["lr"],
                             epsilon=CONFIG["epsilon"])

            epoch_loss += batch_loss(model, X_b, y_b)

        # ── Epoch metrics ──────────────────────────────────────────
        mean_loss  = epoch_loss / n_batches
        train_preds = model.predict_batch(X_train[:200])   # subset for speed
        test_preds  = model.predict_batch(X_test)

        train_acc = accuracy(train_preds, y_train[:200])
        test_acc  = accuracy(test_preds,  y_test)

        history["train_loss"].append(mean_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"  Loss: {mean_loss:.4f} | "
              f"Train acc: {train_acc*100:.1f}% | "
              f"Test acc:  {test_acc*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Final test accuracy: {history['test_acc'][-1]*100:.1f}%")

    # ── Save phases ───────────────────────────────────────────────
    all_phases = [layer.phases for layer in model.layers]
    np.save(CONFIG["save_path"], np.array(all_phases, dtype=object),
            allow_pickle=True)
    print(f"Phases saved to {CONFIG['save_path']}")

    return model, history


if __name__ == "__main__":
    model, history = train()
