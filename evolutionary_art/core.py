"""
Core evolutionary art: expression generation, evolution loop, image saving.

Contains the basic EvolutionaryArtSystem and ArtisticFunction classes
extracted from the original evolutionary-art.py script.
"""

import numpy as np
from PIL import Image
import random
import colorsys
import copy
import os
import re
from deap import base, creator, tools

from evolutionary_art.fitness import evaluate_color_harmony

# A numeric literal in an expression (repr of a float, e.g. -1.25 or 3.5e-05).
COEFFICIENT = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")

# Names an ArtisticFunction expression may use (see _generate_subfunction).
EXPRESSION_NAMES = {"sin": np.sin, "cos": np.cos, "abs": np.abs}


def eval_function(function_str, X, Y):
    """Evaluate an ArtisticFunction expression string on coordinate grids X, Y."""
    namespace = {"x": X, "y": Y, **EXPRESSION_NAMES}
    return eval(function_str, {"__builtins__": {}}, namespace)


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
        """Change the expression: 30% a new random expression, otherwise scale
        coefficients by U(0.5, 1.5) (each with probability 0.3, at least one)."""
        if random.random() < 0.3:
            new = self.generate_random_function()
            while new == self.function_str:
                new = self.generate_random_function()
            self.function_str = new
            return
        matches = list(COEFFICIENT.finditer(self.function_str))
        if not matches:
            self.function_str = self.generate_random_function()
            return
        chosen = [m for m in matches if random.random() < 0.3]
        if not chosen:
            chosen = [random.choice(matches)]
        pieces, last = [], 0
        for m in chosen:
            num = float(m.group())
            new_num = num * random.uniform(0.5, 1.5)
            if new_num == num:
                new_num = random.uniform(-2, 2)
            pieces += [self.function_str[last:m.start()], repr(new_num)]
            last = m.end()
        self.function_str = "".join(pieces) + self.function_str[last:]


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
                        result = eval_function(func.function_str, X, Y)
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
        try:
            img = render_individual(individual, self.width, self.height)
        except Exception:
            return (0,)
        parts = self.fitness_components(img)
        return (parts["fitness"],)

    def fitness_components(self, img):
        """Complexity, balance and colour-harmony terms and their weighted sum."""
        complexity = np.std(img)
        balance = 1 - abs(
            np.mean(img[:, : self.width // 2]) - np.mean(img[:, self.width // 2 :])
        )
        color_harmony = evaluate_color_harmony(img, self.color_palettes)

        fitness = complexity * 0.3 + balance * 0.3 + color_harmony * 0.4
        return {"complexity": complexity, "balance": balance,
                "color_harmony": color_harmony, "fitness": fitness}

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

    def evolve_steps(self):
        """Run the GA, yielding (generation, population, best individual) after each generation."""
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
            offspring = toolbox.select(population, len(population))
            # Deep copies: selection returns the same individual several times, and
            # crossover/mutation must not change genes shared with other offspring.
            offspring = [copy.deepcopy(ind) for ind in offspring]

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
            best_ind = None
            if valid_individuals:
                best_ind = max(valid_individuals, key=lambda x: x.fitness.values[0])
            yield gen, population, best_ind

    def evolve_art(self, output_dir="evolution_output"):
        os.makedirs(output_dir, exist_ok=True)
        for gen, _population, best_ind in self.evolve_steps():
            print(f"\nGeneration {gen}")
            if best_ind is not None:
                print(f"Best Fitness = {best_ind.fitness.values[0]:.3f}")
                save_artwork(
                    best_ind,
                    os.path.join(output_dir, f"generation_{gen}.png"),
                    self.width,
                    self.height,
                )


def render_individual(individual, width=800, height=800):
    """Render an individual (3 ArtisticFunctions -> R, G, B) to a float image in [0, 1]."""
    img = np.zeros((height, width, 3))
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    for i, func in enumerate(individual):
        result = eval_function(func.function_str, X, Y)
        result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-10)
        img[:, :, i] = result
    return img


def save_artwork(individual, filename, width=800, height=800):
    """Render an individual to an image file."""
    img = (render_individual(individual, width, height) * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
