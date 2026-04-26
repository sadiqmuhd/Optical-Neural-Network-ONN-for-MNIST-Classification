"""
evaluate.py
===========
Evaluate a trained ONN on MNIST and generate visualizations:
  1. Confusion matrix
  2. Training loss / accuracy curves
  3. Sample predictions with optical field visualization
  4. Transfer matrix heatmaps (what the optical layers learned)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data.mnist_loader import load_mnist
from onn.network import ONN
from onn.utils import softmax, accuracy


# ── Re-run a quick training for demo (or load saved phases) ───────────────────
def build_demo_model(load_phases_path: str = None) -> ONN:
    model = ONN(input_size=64, num_layers=3, layer_depth=2, num_classes=10)
    if load_phases_path:
        try:
            phases = np.load(load_phases_path, allow_pickle=True)
            for i, layer in enumerate(model.layers):
                layer.phases = phases[i]
            print(f"Loaded phases from {load_phases_path}")
        except FileNotFoundError:
            print("No saved phases found — using random initialization")
    return model


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title("ONN MNIST — Confusion Matrix", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# ── 2. Training Curves ────────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], 'b-o', markersize=4, label='Train Loss')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot([a*100 for a in history["train_acc"]], 'g-o', markersize=4, label='Train Acc')
    axes[1].plot([a*100 for a in history["test_acc"]],  'r-o', markersize=4, label='Test Acc')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Classification Accuracy"); axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Optical Neural Network — Training Progress", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# ── 3. Sample Predictions ─────────────────────────────────────────────────────
def plot_sample_predictions(model: ONN, X_test, y_test,
                             n_samples=10, save_path="sample_predictions.png"):
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 5))

    for col, idx in enumerate(indices):
        x, label = X_test[idx], y_test[idx]

        # Top row: input image
        axes[0, col].imshow(x.reshape(8, 8), cmap='gray')
        axes[0, col].axis('off')
        axes[0, col].set_title(f"True: {label}", fontsize=8)

        # Bottom row: output probabilities
        scores = model.forward(x)
        probs  = softmax(scores)
        pred   = np.argmax(probs)

        color  = 'green' if pred == label else 'red'
        axes[1, col].bar(range(10), probs, color=color, alpha=0.7)
        axes[1, col].set_xticks(range(10))
        axes[1, col].set_xticklabels(range(10), fontsize=7)
        axes[1, col].set_title(f"Pred: {pred}", fontsize=8, color=color)
        axes[1, col].set_ylim(0, 1)
        if col == 0:
            axes[1, col].set_ylabel("Probability", fontsize=8)

    plt.suptitle("ONN Sample Predictions (green=correct, red=wrong)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# ── 4. Optical Layer Visualization ───────────────────────────────────────────
def plot_optical_layers(model: ONN, save_path="optical_layers.png"):
    n_layers = len(model.layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 9))

    for i, layer in enumerate(model.layers):
        W = layer.build_matrix()

        # Amplitude |W|
        amp = np.abs(W)
        im0 = axes[0, i].imshow(amp, cmap='hot', vmin=0, vmax=amp.max())
        axes[0, i].set_title(f"Layer {i+1} — Amplitude |W|", fontsize=11)
        axes[0, i].set_xlabel("Input mode"); axes[0, i].set_ylabel("Output mode")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

        # Phase ∠W
        phase = np.angle(W)
        im1 = axes[1, i].imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1, i].set_title(f"Layer {i+1} — Phase ∠W (rad)", fontsize=11)
        axes[1, i].set_xlabel("Input mode"); axes[1, i].set_ylabel("Output mode")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

    plt.suptitle("Learned Optical Transfer Matrices",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(load_phases_path: str = "onn_trained_phases.npy"):
    print("=" * 60)
    print("  Optical Neural Network — Evaluation")
    print("=" * 60)

    _, _, X_test, y_test = load_mnist(n_train=100, n_test=500)

    model = build_demo_model(load_phases_path)

    print("\nRunning predictions on test set...")
    y_pred = model.predict_batch(X_test)

    acc = accuracy(y_pred, y_test)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(model, X_test, y_test)
    plot_optical_layers(model)


if __name__ == "__main__":
    evaluate()
