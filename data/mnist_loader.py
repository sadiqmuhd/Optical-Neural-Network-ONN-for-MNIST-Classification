"""
data/mnist_loader.py
====================
Load and preprocess MNIST for ONN input.

Key preprocessing steps:
1. Download MNIST via scikit-learn (28×28 grayscale images)
2. Downsample 28×28 → 8×8 to reduce input size to 64
3. Normalize pixel values to [0, 1]
4. Pad/trim to exactly input_size dimensions

Why reduce to 64?
- ONN matrix size scales as O(N²) — large N is expensive to simulate
- 64 dimensions still captures enough structure to classify digits
- Matches the input_size default in our ONN model
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def downsample(images: np.ndarray, target_size: int = 8) -> np.ndarray:
    """
    Downsample 28×28 MNIST images to target_size × target_size.

    Uses simple block averaging (each output pixel = mean of a block
    of input pixels). This is analogous to how optical systems average
    over spatial modes.

    Args:
        images:      array of shape (N, 784)
        target_size: output spatial dimension (default 8 → 8×8 = 64 pixels)

    Returns:
        array of shape (N, target_size²)
    """
    N = images.shape[0]
    imgs = images.reshape(N, 28, 28)

    block = 28 // target_size          # block size per output pixel
    out   = np.zeros((N, target_size, target_size))

    for i in range(target_size):
        for j in range(target_size):
            out[:, i, j] = imgs[
                :,
                i * block:(i + 1) * block,
                j * block:(j + 1) * block
            ].mean(axis=(1, 2))

    return out.reshape(N, target_size * target_size)


def load_mnist(
    n_train:     int = 3000,
    n_test:      int = 500,
    input_size:  int = 64,
    random_seed: int = 42
):
    """
    Load, downsample, and split MNIST dataset.

    Args:
        n_train:    number of training samples
        n_test:     number of test samples
        input_size: ONN input dimension (must be a perfect square or ≤ 784)
        random_seed: for reproducibility

    Returns:
        X_train, y_train, X_test, y_test  (all numpy arrays)
    """
    print("Loading MNIST from scikit-learn (first run downloads ~12MB)...")
    try:
        mnist  = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y   = mnist.data.astype(np.float32), mnist.target.astype(int)
    except Exception as e:
        print(f"  Network unavailable ({type(e).__name__}). Using synthetic MNIST-like data for demo.")
        print("  On your machine, the real MNIST will be downloaded automatically.")
        rng = np.random.default_rng(random_seed)
        N   = max(n_train + n_test + 200, 5000)
        # Simulate structured digit-like patterns per class
        X = np.zeros((N, 784), dtype=np.float32)
        y = np.zeros(N, dtype=int)
        for i in range(N):
            label     = i % 10
            y[i]      = label
            pattern   = np.zeros((28, 28))
            # Each digit class gets a unique structured blob
            cx, cy    = 8 + (label % 5) * 3, 8 + (label // 5) * 3
            for r in range(28):
                for c in range(28):
                    pattern[r, c] = np.exp(-((r - cx)**2 + (c - cy)**2) / (label + 5))
            X[i] = (pattern + rng.normal(0, 0.05, (28, 28))).flatten().clip(0, 1)
        print(f"  Generated {N} synthetic samples.")

    # Normalize to [0, 1]
    X /= 255.0

    # Downsample to 8×8 = 64 dimensions
    target_spatial = int(np.sqrt(input_size))
    if target_spatial * target_spatial != input_size:
        # If input_size isn't a perfect square, use first input_size features
        target_spatial = 8
        input_size     = 64
        print(f"  input_size adjusted to 64 (8×8 downsample)")

    print(f"  Downsampling 28×28 → {target_spatial}×{target_spatial}...")
    X = downsample(X, target_size=target_spatial)

    # Stratified split to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=n_train,
        test_size=n_test,
        stratify=y,
        random_state=random_seed
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Input dimension  : {X_train.shape[1]}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return X_train, y_train, X_test, y_test
