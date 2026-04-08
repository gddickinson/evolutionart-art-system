"""
Image rendering and post-processing for enhanced evolutionary art.

Contains the EnhancedEvolutionaryArt class that creates artwork from
EnhancedArtisticFunction individuals with style-specific effects.
"""

import numpy as np
from PIL import Image
import random
import colorsys
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction
from evolutionary_art.fitness import evaluate_composition_thirds


class EnhancedEvolutionaryArt:
    """Enhanced evolutionary art system with moods, styles, and layer composition."""

    BASE_COLORS = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
    ]

    def __init__(self, width=800, height=800, seed=None, mood=None, style=None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.intent = ArtisticIntent(mood=mood, style=style)
        self.population_size = 20
        self.generations = 50

    def create_artwork(self, functions):
        """Create an image from a list of EnhancedArtisticFunction individuals."""
        img = np.zeros((self.height, self.width, 3))
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)

        theta = random.uniform(0, 2 * np.pi)
        scale = random.uniform(0.5, 2.0)
        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)
        X_final = X_rot * scale
        Y_final = Y_rot * scale

        for func in functions:
            try:
                layers = []
                for _ in range(3):
                    shift_x = random.uniform(-0.5, 0.5)
                    shift_y = random.uniform(-0.5, 0.5)
                    frequency = random.uniform(1, 5)

                    namespace = {
                        "x": X_final * frequency + shift_x,
                        "y": Y_final * frequency + shift_y,
                        "np": np,
                        "self": func,
                        "random": random.random(),
                    }

                    result = eval(
                        func.generate_expression(), {"__builtins__": {}}, namespace
                    )

                    if (
                        isinstance(result, np.ndarray)
                        and result.shape == (self.height, self.width)
                        and np.isfinite(result).all()
                    ):
                        result = (result - np.min(result)) / (
                            np.max(result) - np.min(result) + 1e-6
                        )
                        layers.append(result)

                if layers:
                    combined = self._combine_layers(layers)
                    base_color = random.choice(self.BASE_COLORS)
                    color_variation = np.random.uniform(0.5, 1.5, 3)
                    for c in range(3):
                        img[:, :, c] += combined * base_color[c] * color_variation[c]

            except Exception as e:
                print(f"Error in function evaluation: {e}")
                continue

        if np.max(img) > 0:
            img = img / np.max(img)

        img = self._apply_style_effects(img)
        img = np.clip((img - 0.5) * 1.5 + 0.5, 0, 1)
        return img

    def _combine_layers(self, layers):
        """Combine multiple layers with random blending operations."""
        combined = layers[0]
        for layer in layers[1:]:
            op = random.choice(["add", "multiply", "difference"])
            if op == "add":
                combined = (combined + layer) / 2
            elif op == "multiply":
                combined = combined * layer
            else:
                combined = np.abs(combined - layer)
        return combined

    def _apply_style_effects(self, img):
        """Apply style-specific post-processing effects."""
        if not HAS_CV2:
            return img

        if self.intent.style == "geometric":
            edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
            img = cv2.addWeighted(
                img, 0.7, edges.reshape(self.height, self.width, 1) / 255, 0.3, 0
            )
        elif self.intent.style == "fluid":
            img = cv2.GaussianBlur(img, (15, 15), 0)
            img = cv2.addWeighted(
                img, 1.5, cv2.GaussianBlur(img, (31, 31), 0), -0.5, 0
            )
        elif self.intent.style == "organic":
            noise = np.random.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)
            img = cv2.bilateralFilter(img.astype(np.float32), 9, 75, 75)

        return img

    def evaluate_aesthetics(self, img):
        """Score an image on complexity, composition, and color harmony."""
        intensity_complexity = np.std(img)
        gradient_complexity = np.mean(np.abs(np.gradient(img)))

        edge_complexity = 0
        if HAS_CV2:
            edge_complexity = np.mean(
                cv2.Canny((img * 255).astype(np.uint8), 100, 200)
            )

        complexity = (
            intensity_complexity + gradient_complexity + edge_complexity / 255
        ) / 3
        if self.intent.mood == "energetic":
            complexity *= 1.5

        composition_score = evaluate_composition_thirds(img)

        color_scores = []
        for theme_color in self.intent.color_theme:
            hue, sat, val = theme_color
            rgb = np.array(colorsys.hsv_to_rgb(hue, sat, val))
            color_presence = 1.0 - np.mean(np.abs(img - rgb.reshape(1, 1, 3)))
            color_contrast = np.mean(
                np.abs(np.gradient(np.sum(img * rgb.reshape(1, 1, 3), axis=2)))
            )
            color_scores.append(color_presence + color_contrast)

        color_harmony = np.mean(color_scores)

        style_weights = {
            "organic": [0.4, 0.3, 0.3],
            "geometric": [0.3, 0.4, 0.3],
            "fluid": [0.35, 0.35, 0.3],
        }
        weights = style_weights.get(self.intent.style, [0.33, 0.33, 0.34])

        aesthetic_score = (
            complexity * weights[0]
            + composition_score * weights[1]
            + color_harmony * weights[2]
        )
        aesthetic_score *= random.uniform(0.95, 1.05)
        return aesthetic_score

    def evolve(self, output_dir="evolution_output"):
        """Run the evolutionary loop."""
        os.makedirs(output_dir, exist_ok=True)
        population = [
            EnhancedArtisticFunction(self.intent)
            for _ in range(self.population_size)
        ]
        best_score = float("-inf")
        generations_without_improvement = 0

        for generation in range(self.generations):
            artworks = []
            scores = []

            for individual in population:
                artwork = self.create_artwork([individual])
                score = self.evaluate_aesthetics(artwork)
                artworks.append(artwork)
                scores.append(score)

            current_best = max(scores)
            if current_best > best_score:
                best_score = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            mutation_rate = min(0.8, 0.2 + (generations_without_improvement * 0.1))

            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < 0.7:
                    parents = []
                    for _ in range(2):
                        candidates = random.sample(list(enumerate(population)), 3)
                        parent_idx = max(candidates, key=lambda x: scores[x[0]])[0]
                        parents.append(population[parent_idx])

                    child = EnhancedArtisticFunction(self.intent)
                    child.functions = list(
                        set(parents[0].functions + parents[1].functions)
                    )

                    if random.random() < mutation_rate:
                        extra_funcs = ["sin", "cos", "wave", "flow", "turbulence"]
                        child.functions.extend(
                            random.sample(extra_funcs, k=random.randint(1, 3))
                        )

                    new_population.append(child)
                else:
                    new_population.append(EnhancedArtisticFunction(self.intent))

            num_new = max(1, self.population_size // 10)
            new_population[-num_new:] = [
                EnhancedArtisticFunction(self.intent) for _ in range(num_new)
            ]

            population = new_population

            best_idx = np.argmax(scores)
            best_artwork = artworks[best_idx]
            Image.fromarray((best_artwork * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"generation_{generation}.png")
            )
            print(f"Generation {generation}: Best Score = {scores[best_idx]:.3f}")
            print(f"Mutation Rate: {mutation_rate:.2f}")
