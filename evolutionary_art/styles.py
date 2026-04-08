"""
Artistic style and mood definitions for enhanced evolutionary art.

Contains ArtisticIntent (mood, style, color theme) and
EnhancedArtisticFunction with style-specific math functions.
"""

import numpy as np
import random
import colorsys


class ArtisticIntent:
    """Represents the artistic intent or style of a generation."""

    MOODS = ["calm", "energetic", "mysterious", "harmonious"]
    STYLES = ["geometric", "organic", "fluid", "structured"]

    def __init__(self, mood=None, style=None):
        self.mood = mood or random.choice(self.MOODS)
        self.style = style or random.choice(self.STYLES)
        self.color_theme = self.generate_color_theme()
        self.composition_weight = random.uniform(0.3, 0.7)

    def generate_color_theme(self):
        base_hue = random.random()
        if random.random() < 0.3:
            return [
                (base_hue, 0.7, 0.9),
                ((base_hue + 0.5) % 1.0, 0.7, 0.9),
            ]
        else:
            spread = random.uniform(0.05, 0.15)
            return [
                (base_hue, 0.7, 0.9),
                ((base_hue + spread) % 1.0, 0.7, 0.9),
                ((base_hue - spread) % 1.0, 0.7, 0.9),
            ]


class EnhancedArtisticFunction:
    """Style-aware function generator with mood-specific math operations."""

    def __init__(self, intent):
        self.intent = intent
        self.functions = self._build_function_set()

    def _build_function_set(self):
        funcs = ["sin", "cos", "wave"]

        style_map = {
            "geometric": ["abs", "sin", "cos"],
            "organic": ["sigmoid", "smooth_abs", "flow"],
            "fluid": ["flow", "wave", "turbulence"],
            "structured": ["sin", "cos", "abs"],
        }
        funcs.extend(style_map.get(self.intent.style, []))

        mood_map = {
            "energetic": ["turbulence", "wave"],
            "calm": ["smooth_abs", "flow"],
            "mysterious": ["flow", "wave"],
            "harmonious": ["sin", "cos"],
        }
        funcs.extend(mood_map.get(self.intent.mood, []))

        return list(set(funcs))

    # --- math operations ---

    def sigmoid(self, x, y=None):
        if y is not None:
            return 1 / (1 + np.exp(-(x + y)))
        return 1 / (1 + np.exp(-x))

    def smooth_abs(self, x, y=None):
        if y is not None:
            return np.sqrt(x ** 2 + y ** 2 + 1e-6)
        return np.sqrt(x ** 2 + 1e-6)

    def flow(self, x, y):
        return np.sin(x + np.cos(y)) * np.cos(y + np.sin(x))

    def wave(self, x, y):
        return np.sin(3 * np.sqrt(x ** 2 + y ** 2))

    def turbulence(self, x, y, octaves=3):
        value = 0
        for i in range(octaves):
            freq = 2 ** i
            value += np.sin(x * freq + y * freq) / freq
        return value

    # --- expression building ---

    def generate_expression(self):
        complexity = 3 if self.intent.mood == "energetic" else 2
        return self._build_expression(complexity)

    def _build_expression(self, depth):
        if depth <= 0:
            return f"({random.uniform(-2, 2)} * {random.choice(['x', 'y'])})"

        try:
            func = random.choice(self.functions)
            if func in ["sigmoid", "smooth_abs"]:
                return f"self.{func}({self._build_expression(depth - 1)}, None)"
            elif func in ["flow", "wave", "turbulence"]:
                return f"self.{func}(x, y)"
            else:
                return f"np.{func}({self._build_expression(depth - 1)})"
        except IndexError:
            return f"np.sin({self._build_expression(depth - 1)})"
