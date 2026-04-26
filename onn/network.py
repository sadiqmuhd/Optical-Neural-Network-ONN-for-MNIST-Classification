"""
onn/network.py
==============
Full Optical Neural Network model built from MZI layers.

Architecture:
    Input (complex field)
        ↓
    OpticalLayer 1  (unitary matrix built from MZI grid)
        ↓
    OpticalLayer 2
        ↓
    ...
        ↓
    OpticalLayer N
        ↓
    Photodetection  |y|²  (complex → real intensity)
        ↓
    Softmax → class probabilities (0–9)
"""

import numpy as np
from .layers import mzi, random_unitary
from .utils import photodetect, softmax, cross_entropy_loss


class OpticalLayer:
    """
    One optical layer = an N×N unitary matrix implemented as a grid of MZIs.

    For an N-port device, we use a triangular MZI mesh:
    - N//2 MZIs acting on pairs (0,1), (2,3), ... in even columns
    - N//2 MZIs acting on pairs (1,2), (3,4), ... in odd columns
    - Repeated for depth passes to approximate any unitary matrix

    The learnable parameters are the phase angles of each MZI.
    """

    def __init__(self, size: int, depth: int = 2):
        """
        Args:
            size:  number of optical modes (waveguide channels)
            depth: number of MZI columns (more = more expressive)
        """
        self.size  = size
        self.depth = depth
        # phases[col][mzi_index] = [phi1, phi2]
        # Each MZI has 2 trainable phases
        num_mzis_per_col = size // 2
        self.phases = np.random.uniform(
            0, 2 * np.pi,
            (depth, num_mzis_per_col, 2)
        )

    def build_matrix(self) -> np.ndarray:
        """
        Construct the full N×N transfer matrix from current phase values.

        Process:
        - Start with identity
        - Apply each column of MZIs as 2×2 blocks
        - Alternate even/odd pairing between columns
        """
        W = np.eye(self.size, dtype=complex)

        for col in range(self.depth):
            col_matrix = np.eye(self.size, dtype=complex)
            # Even columns: pair (0,1), (2,3), ...
            # Odd columns:  pair (1,2), (3,4), ...
            offset = col % 2
            mzi_idx = 0
            for i in range(offset, self.size - 1, 2):
                phi1, phi2 = self.phases[col, mzi_idx % (self.size // 2)]
                M = mzi(phi1, phi2)
                col_matrix[i:i+2, i:i+2] = M
                mzi_idx += 1
            W = col_matrix @ W

        return W

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagate optical field x through this layer.

        Args:
            x: complex field vector of shape (size,)

        Returns:
            complex field vector of shape (size,)
        """
        W = self.build_matrix()
        return W @ x

    def get_phases(self) -> np.ndarray:
        """Return flat array of all phase parameters."""
        return self.phases.flatten()

    def set_phases(self, flat_phases: np.ndarray):
        """Set phases from flat array (used during gradient updates)."""
        self.phases = flat_phases.reshape(self.phases.shape)


class ONN:
    """
    Full Optical Neural Network.

    Stacks multiple OpticalLayers and finishes with photodetection.
    Trained using finite-difference gradient estimation (zeroth-order
    optimization) — appropriate since physical ONNs can't backpropagate
    through hardware.
    """

    def __init__(self,
                 input_size:  int = 64,
                 num_layers:  int = 3,
                 layer_depth: int = 2,
                 num_classes: int = 10):
        """
        Args:
            input_size:  dimensionality of input (must be even)
            num_layers:  number of optical layers to stack
            layer_depth: MZI columns per layer (expressiveness)
            num_classes: output classes (10 for MNIST digits)
        """
        assert input_size % 2 == 0, "input_size must be even (pairs of waveguides)"
        assert input_size >= num_classes, "input_size must be >= num_classes"

        self.input_size  = input_size
        self.num_classes = num_classes

        self.layers = [
            OpticalLayer(size=input_size, depth=layer_depth)
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Full forward pass through the ONN.

        Steps:
        1. Cast input to complex (real pixel values become complex amplitudes)
        2. Propagate through each optical layer
        3. Photodetect (|y|² — measure intensity)
        4. Take first num_classes outputs as logits

        Args:
            x: real input vector of shape (input_size,)

        Returns:
            real intensity vector of shape (num_classes,) — raw class scores
        """
        # Step 1: encode input as complex optical field
        field = x.astype(complex)

        # Step 2: propagate through optical layers
        for layer in self.layers:
            field = layer.forward(field)

        # Step 3: photodetection — |E|² gives intensity
        intensity = photodetect(field)

        # Step 4: select output channels for classification
        return intensity[:self.num_classes]

    def predict(self, x: np.ndarray) -> int:
        """
        Predict class label for a single input.

        Args:
            x: input vector

        Returns:
            predicted class index (0–9)
        """
        scores = self.forward(x)
        probs  = softmax(scores)
        return int(np.argmax(probs))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of inputs.

        Args:
            X: input matrix of shape (N, input_size)

        Returns:
            array of predicted class indices, shape (N,)
        """
        return np.array([self.predict(x) for x in X])

    def count_parameters(self) -> int:
        """Return total number of trainable phase parameters."""
        total = 0
        for layer in self.layers:
            total += layer.phases.size
        return total
