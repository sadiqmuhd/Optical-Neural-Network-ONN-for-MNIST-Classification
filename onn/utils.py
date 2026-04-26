"""
onn/utils.py
============
Helper functions for optical detection and loss computation.
"""

import numpy as np


def photodetect(field: np.ndarray) -> np.ndarray:
    """
    Simulate photodetection: convert complex optical field to intensity.

    Physically: a photodetector measures optical power |E|²,
    losing phase information — only amplitude survives.

    Args:
        field: complex-valued optical field vector

    Returns:
        real-valued intensity vector (same shape)
    """
    return np.abs(field) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax — converts raw scores to probabilities.

    Args:
        x: real-valued score vector

    Returns:
        probability vector summing to 1
    """
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-12)


def cross_entropy_loss(probs: np.ndarray, label: int) -> float:
    """
    Cross-entropy loss for a single sample.

    Args:
        probs: probability vector from softmax
        label: true class index (0–9 for MNIST)

    Returns:
        scalar loss value
    """
    return -np.log(probs[label] + 1e-9)


def one_hot(label: int, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer label to one-hot vector.

    Args:
        label: integer class label
        num_classes: total number of classes

    Returns:
        one-hot encoded numpy array
    """
    vec = np.zeros(num_classes)
    vec[label] = 1.0
    return vec


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: array of predicted class indices
        labels: array of true class indices

    Returns:
        accuracy as a float between 0 and 1
    """
    return np.mean(predictions == labels)
