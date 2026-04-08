"""
Fitness evaluation functions for evolutionary art.

Provides aesthetic scoring, color harmony evaluation, and composition
analysis used by both the basic and enhanced art systems.
"""

import numpy as np


def evaluate_color_harmony(img, color_palettes):
    """
    Evaluate how well an image's colors match harmonious palettes.

    Args:
        img: numpy array (H, W, 3) with values in [0, 1].
        color_palettes: list of palettes, each a list of [R, G, B] (0-255).

    Returns:
        float: harmony score (higher is better).
    """
    pixels = img.reshape(-1, 3)

    max_harmony = 0
    for palette in color_palettes:
        palette_norm = np.array(palette) / 255.0
        harmony = 0
        for color in palette_norm:
            distances = np.linalg.norm(pixels - color, axis=1)
            harmony += np.exp(-np.min(distances) * 10)
        max_harmony = max(max_harmony, harmony / len(palette))

    return max_harmony


def evaluate_complexity(img):
    """Measure image complexity via standard deviation."""
    return np.std(img)


def evaluate_balance(img):
    """Measure left-right balance of an image."""
    w = img.shape[1]
    return 1 - abs(np.mean(img[:, : w // 2]) - np.mean(img[:, w // 2 :]))


def evaluate_composition_thirds(img):
    """
    Evaluate interest at rule-of-thirds points.

    Args:
        img: numpy array (H, W, 3) with values in [0, 1].

    Returns:
        float: composition score.
    """
    h, w = img.shape[:2]
    third_h = h // 3
    third_w = w // 3
    thirds_points = [
        (third_w, third_h), (2 * third_w, third_h),
        (third_w, 2 * third_h), (2 * third_w, 2 * third_h),
    ]

    score = 0
    for px, py in thirds_points:
        y_lo = max(0, py - 10)
        y_hi = min(h, py + 10)
        x_lo = max(0, px - 10)
        x_hi = min(w, px + 10)
        region = img[y_lo:y_hi, x_lo:x_hi]
        score += np.std(region)

    return score / len(thirds_points)
