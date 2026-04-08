import numpy as np
from PIL import Image
import random
import math
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import colorsys

class ArtisticFunction:
    def __init__(self):
        self.function_str = self.generate_random_function()
        
    def generate_random_function(self):
        depth = random.randint(2, 4)
        return self._generate_subfunction(depth)
    
    def _generate_subfunction(self, depth):
        if depth <= 1:
            var = random.choice(['x', 'y'])
            coef = random.uniform(-2, 2)
            return f"({coef} * {var})"
        
        if random.random() < 0.4:  # 40% chance for special function
            func = random.choice(['sin', 'cos'])
            inner = self._generate_subfunction(depth-1)
            return f"{func}({inner})"
        else:  # 60% chance for basic arithmetic
            op = random.choice(['+', '*'])
            left = self._generate_subfunction(depth-1)
            right = self._generate_subfunction(depth-1)
            return f"({left} {op} {right})"
    
    def mutate(self):
        if random.random() < 0.3:
            self.function_str = self.generate_random_function()
        else:
            # Simple mutation: change a coefficient
            parts = self.function_str.split('*')
            for i, part in enumerate(parts):
                if random.random() < 0.3 and '.' in part:  # If part contains a number
                    try:
                        # Extract number and modify it
                        num = float(part.split()[0].strip('() '))
                        new_num = num * random.uniform(0.5, 1.5)
                        self.function_str = self.function_str.replace(str(num), str(new_num))
                    except:
                        pass  # Skip if parsing fails

class EvolutionaryArtSystem:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.population_size = 20
        self.generations = 50
        self.artistic_functions = []
        self.color_palettes = self.generate_color_palettes()
        
    def generate_color_palettes(self):
        palettes = []
        # Generate harmonious color palettes
        for _ in range(5):
            base_hue = random.random()
            palette = []
            # Analogous colors
            for i in range(3):
                hue = (base_hue + 0.1 * i) % 1.0
                sat = random.uniform(0.5, 1.0)
                val = random.uniform(0.5, 1.0)
                rgb = colorsys.hsv_to_rgb(hue, sat, val)
                palette.append([int(c * 255) for c in rgb])
            palettes.append(palette)
        return palettes
    
    def initialize_population(self):
        self.artistic_functions = []
        
        # Keep trying until we get enough valid individuals
        while len(self.artistic_functions) < self.population_size:
            try:
                # Create new individual
                individual = [ArtisticFunction() for _ in range(3)]
                
                # Test if it produces valid output
                x = np.linspace(-1, 1, 10)  # Small test grid
                y = np.linspace(-1, 1, 10)
                X, Y = np.meshgrid(x, y)
                
                valid = True
                for func in individual:
                    try:
                        namespace = {
                            'x': X, 
                            'y': Y,
                            'sin': np.sin,
                            'cos': np.cos,
                            'abs': np.abs
                        }
                        result = eval(func.function_str, {"__builtins__": {}}, namespace)
                        if not np.isfinite(result).all():
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if valid:
                    self.artistic_functions.append(individual)
                    print(f"Added valid individual {len(self.artistic_functions)}/{self.population_size}")
                
            except Exception as e:
                print(f"Error creating individual: {str(e)}")
    
    def evaluate_individual(self, individual):
        # Create the image
        img = np.zeros((self.height, self.width, 3))
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate each function for RGB channels
        for i, func in enumerate(individual):
            try:
                # Safe eval with limited namespace
                namespace = {'x': X, 'y': Y, 'np': np}
                result = eval(func.function_str, {"__builtins__": {}}, namespace)
                
                # Normalize to [0, 1]
                result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-10)
                img[:, :, i] = result
                
            except:
                return 0,  # Return low fitness if evaluation fails
        
        # Calculate aesthetic measures
        complexity = np.std(img)  # Measure of detail
        balance = 1 - abs(np.mean(img[:, :self.width//2]) - np.mean(img[:, self.width//2:]))
        color_harmony = self.evaluate_color_harmony(img)
        
        fitness = (complexity * 0.3 + balance * 0.3 + color_harmony * 0.4)
        return fitness,
    
    def evaluate_color_harmony(self, img):
        # Calculate color distribution and compare with harmonious palettes
        pixels = img.reshape(-1, 3)
        hist, _ = np.histogramdd(pixels, bins=8, range=((0,1), (0,1), (0,1)))
        hist = hist / hist.sum()
        
        # Compare with color palettes
        max_harmony = 0
        for palette in self.color_palettes:
            palette_norm = np.array(palette) / 255.0
            harmony = 0
            for color in palette_norm:
                # Find closest colors in image
                distances = np.linalg.norm(pixels - color, axis=1)
                harmony += np.exp(-np.min(distances) * 10)
            max_harmony = max(max_harmony, harmony / len(palette))
        
        return max_harmony
    
    def crossover(self, ind1, ind2):
        # Crossover functions between individuals
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i].function_str, ind2[i].function_str = \
                    ind2[i].function_str, ind1[i].function_str
        return ind1, ind2
    
    def mutate(self, individual):
        # Mutate each function in the individual
        for func in individual:
            if random.random() < 0.2:
                func.mutate()
        return individual,
    
    def evolve_art(self, output_dir="evolution_output"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize population
        self.initialize_population()
        
        # Setup genetic algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population with proper Individual wrapper
        population = []
        for funcs in self.artistic_functions:
            ind = creator.Individual(funcs)
            # Evaluate and assign initial fitness
            ind.fitness.values = self.evaluate_individual(ind)
            population.append(ind)
        
        # Evolution loop
        for gen in range(self.generations):
            print(f"\nGeneration {gen}")
            
            # Select next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(creator.Individual, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate all invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)
            
            # Replace population
            population[:] = offspring
            
            # Find best individual
            valid_individuals = [ind for ind in population if ind.fitness.valid]
            if valid_individuals:
                best_ind = max(valid_individuals, key=lambda x: x.fitness.values[0])
                print(f"Best Fitness = {best_ind.fitness.values[0]:.3f}")
                self.save_artwork(best_ind, os.path.join(output_dir, f"generation_{gen}.png"))
            else:
                print("No valid individuals in this generation")
    
    def save_artwork(self, individual, filename):
        img = np.zeros((self.height, self.width, 3))
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        for i, func in enumerate(individual):
            namespace = {'x': X, 'y': Y, 'sin': np.sin, 'cos': np.cos, 'abs': np.abs}
            result = eval(func.function_str, {"__builtins__": {}}, namespace)
            result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-10)
            img[:, :, i] = result
        
        # Apply color palette enhancement
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(filename)

if __name__ == "__main__":
    # Create and run the evolutionary art system
    art_system = EvolutionaryArtSystem(width=800, height=800)
    art_system.evolve_art()
    print("Evolution complete! Check the evolution_output directory for results.")