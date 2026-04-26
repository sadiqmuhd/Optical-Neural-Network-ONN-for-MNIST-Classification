"""
onn/layers.py
=============
Transfer matrix building blocks for Optical Neural Networks.

Physics background:
- Light is represented as complex numbers (amplitude + phase)
- Each optical component transforms light via matrix multiplication
- Mach-Zehnder Interferometer (MZI) = phase shifter → beamsplitter → phase shifter
- MZIs are the trainable units — we tune their phase angles during training
"""

import numpy as np


def phase_shifter(phi: float) -> np.ndarray:
    """
    Single phase shifter acting on one waveguide channel.

    Physically: applies a phase delay phi to the first channel,
    leaves the second unchanged.

    Args:
        phi: phase angle in radians (this is what gets trained)

    Returns:
        2x2 complex unitary matrix
    """
    return np.array(
        [[np.exp(1j * phi), 0],
         [0,                1]],
        dtype=complex
    )


def beamsplitter() -> np.ndarray:
    """
    50/50 directional coupler (beamsplitter).

    Physically: splits incoming light equally between two waveguides,
    introducing a π/2 phase shift on the cross-coupled output.

    This is a fixed component — not trainable.

    Returns:
        2x2 complex unitary matrix
    """
    return (1.0 / np.sqrt(2)) * np.array(
        [[1,  1j],
         [1j, 1]],
        dtype=complex
    )


def mzi(phi1: float, phi2: float) -> np.ndarray:
    """
    Mach-Zehnder Interferometer (MZI).

    Structure: PhaseShifter(phi1) → BeamSplitter → PhaseShifter(phi2)

    This is the fundamental trainable unit in a physical ONN chip.
    By tuning phi1 and phi2, we control how light is routed between
    two waveguide channels — equivalent to a 2D rotation in complex space.

    Args:
        phi1: input phase shift (radians)
        phi2: output phase shift (radians)

    Returns:
        2x2 complex unitary matrix representing the full MZI
    """
    PS1 = phase_shifter(phi1)
    BS  = beamsplitter()
    PS2 = phase_shifter(phi2)
    return PS2 @ BS @ PS1


def random_unitary(size: int) -> np.ndarray:
    """
    Generate a random unitary matrix of given size using QR decomposition.

    Used to initialize optical layers — a random unitary is a physically
    valid starting point (preserves optical power).

    Args:
        size: matrix dimension

    Returns:
        size x size complex unitary matrix
    """
    # Random complex matrix
    Z = (np.random.randn(size, size) + 1j * np.random.randn(size, size)) / np.sqrt(2)
    # QR decomposition gives orthogonal Q (unitary in complex case)
    Q, R = np.linalg.qr(Z)
    # Fix phase to ensure true Haar-random unitary
    D = np.diag(R) / np.abs(np.diag(R))
    return Q * D
