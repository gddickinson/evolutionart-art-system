"""
Core evolutionary art: expression generation, evolution loop, image saving.

Contains the basic EvolutionaryArtSystem and ArtisticFunction classes
extracted from the original evolutionary-art.py script.
"""

import numpy as np
from PIL import Image
import random
import colorsys
import os
from deap import base, creator, tools

from evolutionary_art.fitness import evaluate_color_harmony


class ArtisticFunction:
    """Generates and mutates mathematical expressions for art."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.function_str = self.generate_random_function()

    def generate_random_function(self):
        depth = random.randint(2, 4)
        return self._generate_subfunction(depth)

    def _generate_subfunction(self, depth):
        if depth <= 1:
            var = random.choice(["x", "y"])
            coef = random.uniform(-2, 2)
            return f"({coef} * {var})"

        if random.random() < 0.4:
            func = random.choice(["sin", "cos"])
            inner = self._generate_subfunction(depth - 1)
            return f"{func}({inner})"
        else:
            op = random.choice(["+", "*"])
            left = self._generate_subfunction(depth - 1)
            right = self._generate_subfunction(depth - 1)
            return f"({left} {op} {right})"

    def mutate(self):
        if random.random() < 0.3:
            self.function_str = self.generate_random_function()
        else:
            parts = self.function_str.split("*")
            for i, part in enumerate(parts):
                if random.random() < 0.3 and "." in part:
                    try:
                        num = float(part.split()[0].strip("() "))
                        new_num = num * random.uniform(0.5, 1.5)
                        self.function_str = self.function_str.replace(
                            str(num), str(new_num)
                        )
                    except (ValueError, IndexError):
                        pass


def generate_color_palettes(n=5):
    """Generate n harmonious color palettes."""
    palettes = []
    for _ in range(n):
        base_hue = random.random()
        palette = []
        for i in range(3):
            hue = (base_hue + 0.1 * i) % 1.0
            sat = random.uniform(0.5, 1.0)
            val = random.uniform(0.5, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            palette.append([int(c * 255) for c in rgb])
        palettes.append(palette)
    return palettes


class EvolutionaryArtSystem:
    """Basic evolutionary art system using genetic algorithms."""

    def __init__(self, width=800, height=800, seed=None):
        self.width = width
        self.height = height
        self.population_size = 20
        self.generations = 50
        self.artistic_functions = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.color_palettes = generate_color_palettes()

    def initialize_population(self):
        self.artistic_functions = []
        while len(self.artistic_functions) < self.population_size:
            try:
                individual = [ArtisticFunction() for _ in range(3)]
                x = np.linspace(-1, 1, 10)
                y = np.linspace(-1, 1, 10)
                X, Y = np.meshgrid(x, y)

                valid = True
                for func in individual:
                    try:
                        namespace = {
                            "x": X, "y": Y,
                            "sin": np.sin, "cos": np.cos, "abs": np.abs,
                        }
                        result = eval(
                            func.function_str, {"__builtins__": {}}, namespace
                        )
                        if not np.isfinite(result).all():
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break

                if valid:
                    self.artistic_functions.append(individual)
            except Exception as e:
                print(f"Error creating individual: {e}")

    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual (list of 3 ArtisticFunctions)."""
        img = np.zeros((self.height, self.width, 3))
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)

        for i, func in enumerate(individual):
            try:
                namespace = {"x": X, "y": Y, "np": np}
                result = eval(func.function_str, {"__builtins__": {}}, namespace)
                result = (result - np.min(result)) / (
                    np.max(result) - np.min(result) + 1e-10
                )
                img[:, :, i] = result
            except Exception:
                return (0,)

        complexity = np.std(img)
        balance = 1 - abs(
            np.mean(img[:, : self.width // 2]) - np.mean(img[:, self.width // 2 :])
        )
        color_harmony = evaluate_color_harmony(img, self.color_palettes)

        fitness = complexity * 0.3 + balance * 0.3 + color_harmony * 0.4
        return (fitness,)

    def crossover(self, ind1, ind2):
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i].function_str, ind2[i].function_str = (
                    ind2[i].function_str,
                    ind1[i].function_str,
                )
        return ind1, ind2

    def mutate_individual(self, individual):
        for func in individual:
            if random.random() < 0.2:
                func.mutate()
        return (individual,)

    def evolve_art(self, output_dir="evolution_output"):
        os.makedirs(output_dir, exist_ok=True)
        self.initialize_population()

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = []
        for funcs in self.artistic_functions:
            ind = creator.Individual(funcs)
            ind.fitness.values = self.evaluate_individual(ind)
            population.append(ind)

        for gen in range(self.generations):
            print(f"\nGeneration {gen}")
            offspring = toolbox.select(population, len(population))
            offspring = list(map(creator.Individual, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)

            population[:] = offspring
            valid_individuals = [ind for ind in population if ind.fitness.valid]
            if valid_individuals:
                best_ind = max(valid_individuals, key=lambda x: x.fitness.values[0])
                print(f"Best Fitness = {best_ind.fitness.values[0]:.3f}")
                save_artwork(
                    best_ind,
                    os.path.join(output_dir, f"generation_{gen}.png"),
                    self.width,
                    self.height,
                )


def save_artwork(individual, filename, width=800, height=800):
    """Render an individual to an image file."""
    img = np.zeros((height, width, 3))
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    for i, func in enumerate(individual):
        namespace = {"x": X, "y": Y, "sin": np.sin, "cos": np.cos, "abs": np.abs}
        result = eval(func.function_str, {"__builtins__": {}}, namespace)
        result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-10)
        img[:, :, i] = result

    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
