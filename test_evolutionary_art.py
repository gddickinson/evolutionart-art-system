"""
Smoke tests for the evolutionary art system.
"""

import pytest
import numpy as np

from evolutionary_art.core import ArtisticFunction, generate_color_palettes, save_artwork
from evolutionary_art.fitness import (
    evaluate_color_harmony,
    evaluate_complexity,
    evaluate_balance,
    evaluate_composition_thirds,
)
from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction
from evolutionary_art.cli import main as cli_main


class TestArtisticFunction:
    def test_generates_function_string(self):
        func = ArtisticFunction(seed=42)
        assert isinstance(func.function_str, str)
        assert len(func.function_str) > 0

    def test_function_evaluates(self):
        func = ArtisticFunction(seed=42)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        namespace = {
            "x": X, "y": Y,
            "sin": np.sin, "cos": np.cos, "abs": np.abs,
        }
        result = eval(func.function_str, {"__builtins__": {}}, namespace)
        assert isinstance(result, np.ndarray)

    def test_mutate_does_not_crash(self):
        func = ArtisticFunction(seed=42)
        original = func.function_str
        func.mutate()
        # Should still be a string after mutation
        assert isinstance(func.function_str, str)


class TestFitness:
    def test_evaluate_color_harmony(self):
        img = np.random.rand(10, 10, 3)
        palettes = generate_color_palettes(3)
        score = evaluate_color_harmony(img, palettes)
        assert isinstance(score, float)
        assert score >= 0

    def test_evaluate_complexity(self):
        img = np.random.rand(10, 10, 3)
        score = evaluate_complexity(img)
        assert score > 0

    def test_evaluate_balance_uniform(self):
        img = np.ones((10, 10, 3)) * 0.5
        assert evaluate_balance(img) == pytest.approx(1.0)

    def test_evaluate_composition_thirds(self):
        img = np.random.rand(100, 100, 3)
        score = evaluate_composition_thirds(img)
        assert isinstance(score, float)


class TestStyles:
    def test_artistic_intent_creation(self):
        intent = ArtisticIntent()
        assert intent.mood in ArtisticIntent.MOODS
        assert intent.style in ArtisticIntent.STYLES
        assert len(intent.color_theme) >= 2

    def test_artistic_intent_fixed(self):
        intent = ArtisticIntent(mood="calm", style="geometric")
        assert intent.mood == "calm"
        assert intent.style == "geometric"

    def test_enhanced_function_generates_expression(self):
        intent = ArtisticIntent(mood="calm", style="organic")
        func = EnhancedArtisticFunction(intent)
        expr = func.generate_expression()
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_enhanced_function_math_operations(self):
        intent = ArtisticIntent()
        func = EnhancedArtisticFunction(intent)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)

        assert func.wave(X, Y).shape == (10, 10)
        assert func.flow(X, Y).shape == (10, 10)
        assert func.turbulence(X, Y).shape == (10, 10)
        assert np.isfinite(func.sigmoid(X)).all()
        assert np.isfinite(func.smooth_abs(X)).all()


class TestCLI:
    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
