import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import colorsys
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2

class ArtisticIntent:
    """Represents the 'artistic intent' or 'style' of the generation"""
    def __init__(self):
        # Core artistic elements
        self.mood = random.choice(['calm', 'energetic', 'mysterious', 'harmonious'])
        self.style = random.choice(['geometric', 'organic', 'fluid', 'structured'])
        self.color_theme = self.generate_color_theme()
        self.composition_weight = random.uniform(0.3, 0.7)
        
    def generate_color_theme(self):
        base_hue = random.random()
        if random.random() < 0.3:  # Complementary colors
            return [(base_hue, 0.7, 0.9), 
                   ((base_hue + 0.5) % 1.0, 0.7, 0.9)]
        else:  # Analogous colors
            spread = random.uniform(0.05, 0.15)
            return [(base_hue, 0.7, 0.9),
                   ((base_hue + spread) % 1.0, 0.7, 0.9),
                   ((base_hue - spread) % 1.0, 0.7, 0.9)]

class EnhancedArtisticFunction:
    def __init__(self, intent):
        self.intent = intent
        self.functions = self.generate_function_set()
        
    def generate_function_set(self):
        # Base functions always available
        funcs = ['sin', 'cos', 'wave']
        
        # Style-specific functions
        if self.intent.style == 'geometric':
            funcs.extend(['abs', 'sin', 'cos'])
        elif self.intent.style == 'organic':
            funcs.extend(['sigmoid', 'smooth_abs', 'flow'])
        elif self.intent.style == 'fluid':
            funcs.extend(['flow', 'wave', 'turbulence'])
        elif self.intent.style == 'structured':
            funcs.extend(['sin', 'cos', 'abs'])
        
        # Mood-specific functions
        if self.intent.mood == 'energetic':
            funcs.extend(['turbulence', 'wave'])
        elif self.intent.mood == 'calm':
            funcs.extend(['smooth_abs', 'flow'])
        elif self.intent.mood == 'mysterious':
            funcs.extend(['flow', 'wave'])
        elif self.intent.mood == 'harmonious':
            funcs.extend(['sin', 'cos'])
            
        return list(set(funcs))  # Remove duplicates
    
    def sigmoid(self, x, y=None):
        if y is not None:
            return 1 / (1 + np.exp(-(x + y)))
        return 1 / (1 + np.exp(-x))
    
    def smooth_abs(self, x, y=None):
        if y is not None:
            return np.sqrt(x**2 + y**2 + 1e-6)
        return np.sqrt(x**2 + 1e-6)
    
    def flow(self, x, y):
        return np.sin(x + np.cos(y)) * np.cos(y + np.sin(x))
    
    def wave(self, x, y):
        return np.sin(3 * np.sqrt(x**2 + y**2))
    
    def turbulence(self, x, y, octaves=3):
        value = 0
        for i in range(octaves):
            freq = 2**i
            value += np.sin(x*freq + y*freq) / freq
        return value
    
    def generate_expression(self):
        complexity = 3 if self.intent.mood == 'energetic' else 2
        return self._build_expression(complexity)
    
    def _build_expression(self, depth):
        if depth <= 0:
            return f"({random.uniform(-2, 2)} * {random.choice(['x', 'y'])})"
        
        try:
            func = random.choice(self.functions)
            if func in ['sigmoid', 'smooth_abs']:
                return f"self.{func}({self._build_expression(depth-1)}, None)"
            elif func in ['flow', 'wave', 'turbulence']:
                return f"self.{func}(x, y)"
            else:
                return f"np.{func}({self._build_expression(depth-1)})"
        except IndexError:
            # Fallback to basic function if selection fails
            return f"np.sin({self._build_expression(depth-1)})"

class EnhancedEvolutionaryArt:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.intent = ArtisticIntent()
        self.population_size = 20
        self.generations = 50
        
    def create_artwork(self, functions):
        img = np.zeros((self.height, self.width, 3))
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Add rotation and scaling for more variation
        theta = random.uniform(0, 2 * np.pi)
        scale = random.uniform(0.5, 2.0)
        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)
        X_final = X_rot * scale
        Y_final = Y_rot * scale
        
        # Apply dramatic color variations
        base_colors = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1),  # Cyan
        ]
        
        for i, func in enumerate(functions):
            try:
                # Create multiple layers with different transformations
                layers = []
                for _ in range(3):  # Create 3 layers per function
                    # Vary the coordinate system for each layer
                    shift_x = random.uniform(-0.5, 0.5)
                    shift_y = random.uniform(-0.5, 0.5)
                    frequency = random.uniform(1, 5)
                    
                    namespace = {
                        'x': X_final * frequency + shift_x,
                        'y': Y_final * frequency + shift_y,
                        'np': np,
                        'self': func,
                        'random': random.random()
                    }
                    
                    result = eval(func.generate_expression(), {"__builtins__": {}}, namespace)
                    
                    if isinstance(result, np.ndarray) and result.shape == (self.height, self.width):
                        if np.isfinite(result).all():
                            # Normalize the layer
                            result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-6)
                            layers.append(result)
                
                if layers:
                    # Combine layers with different operations
                    combined = layers[0]
                    for layer in layers[1:]:
                        op = random.choice(['add', 'multiply', 'difference'])
                        if op == 'add':
                            combined = (combined + layer) / 2
                        elif op == 'multiply':
                            combined = combined * layer
                        else:
                            combined = np.abs(combined - layer)
                    
                    # Apply color with more dramatic variations
                    base_color = random.choice(base_colors)
                    color_variation = np.random.uniform(0.5, 1.5, 3)
                    for c in range(3):
                        img[:, :, c] += combined * base_color[c] * color_variation[c]
                
            except Exception as e:
                print(f"Error in function evaluation: {str(e)}")
                continue
        
        # Normalize and apply dramatic post-processing
        if np.max(img) > 0:
            img = img / np.max(img)
        
        # Apply style-specific effects
        if self.intent.style == 'geometric':
            # Enhance edges
            edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
            img = cv2.addWeighted(img, 0.7, edges.reshape(self.height, self.width, 1)/255, 0.3, 0)
        elif self.intent.style == 'fluid':
            # Add flow-like effects
            img = cv2.GaussianBlur(img, (15, 15), 0)
            img = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (31, 31), 0), -0.5, 0)
        elif self.intent.style == 'organic':
            # Add texture
            noise = np.random.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)
            img = cv2.bilateralFilter(img.astype(np.float32), 9, 75, 75)
        
        # Final contrast enhancement
        img = np.clip((img - 0.5) * 1.5 + 0.5, 0, 1)
        
        return img
    
    def evaluate_aesthetics(self, img):
        # Calculate complexity using multiple measures
        intensity_complexity = np.std(img)
        gradient_complexity = np.mean(np.abs(np.gradient(img)))
        edge_complexity = np.mean(cv2.Canny(
            (img * 255).astype(np.uint8), 100, 200
        ))
        
        complexity = (intensity_complexity + gradient_complexity + edge_complexity/255) / 3
        complexity *= (1.5 if self.intent.mood == 'energetic' else 1.0)
        
        # Enhanced composition evaluation
        h, w = img.shape[:2]
        
        # Rule of thirds points
        third_h = h // 3
        third_w = w // 3
        thirds_points = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        
        # Evaluate interest at rule of thirds points
        composition_score = 0
        for px, py in thirds_points:
            region = img[py-10:py+10, px-10:px+10]
            local_contrast = np.std(region)
            local_edges = np.mean(cv2.Canny(
                (region * 255).astype(np.uint8), 100, 200
            ))
            composition_score += local_contrast + local_edges/255
        
        composition_score = composition_score / len(thirds_points)
        
        # Color harmony evaluation
        color_scores = []
        for theme_color in self.intent.color_theme:
            hue, sat, val = theme_color
            rgb = np.array(colorsys.hsv_to_rgb(hue, sat, val))
            
            # Color presence
            color_presence = 1.0 - np.mean(np.abs(img - rgb.reshape(1, 1, 3)))
            
            # Color contrast
            color_contrast = np.mean(np.abs(np.gradient(
                np.sum(img * rgb.reshape(1, 1, 3), axis=2)
            )))
            
            color_scores.append(color_presence + color_contrast)
        
        color_harmony = np.mean(color_scores)
        
        # Calculate final score with style-specific weights
        if self.intent.style == 'organic':
            weights = [0.4, 0.3, 0.3]  # Favor complexity
        elif self.intent.style == 'geometric':
            weights = [0.3, 0.4, 0.3]  # Favor composition
        elif self.intent.style == 'fluid':
            weights = [0.35, 0.35, 0.3]  # Balance complexity and composition
        else:
            weights = [0.33, 0.33, 0.34]  # Balanced
        
        aesthetic_score = (
            complexity * weights[0] +
            composition_score * weights[1] +
            color_harmony * weights[2]
        )
        
        # Add random variation to break ties and encourage exploration
        aesthetic_score *= random.uniform(0.95, 1.05)
        
        return aesthetic_score
    
    def evolve(self, output_dir="evolution_output"):
        population = [EnhancedArtisticFunction(self.intent) for _ in range(self.population_size)]
        best_score = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            # Create and evaluate artworks
            artworks = []
            scores = []
            
            for individual in population:
                artwork = self.create_artwork([individual])
                score = self.evaluate_aesthetics(artwork)
                artworks.append(artwork)
                scores.append(score)
            
            # Track improvement
            current_best = max(scores)
            if current_best > best_score:
                best_score = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Increase mutation rate if stuck
            mutation_rate = min(0.8, 0.2 + (generations_without_improvement * 0.1))
            
            # Selection and reproduction with enhanced variation
            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < 0.7:  # 70% chance of crossover
                    # Tournament selection
                    parents = []
                    for _ in range(2):
                        candidates = random.sample(list(enumerate(population)), 3)
                        parent_idx = max(candidates, key=lambda x: scores[x[0]])[0]
                        parents.append(population[parent_idx])
                    
                    # Create child with mixed properties
                    child = EnhancedArtisticFunction(self.intent)
                    child.functions = list(set(parents[0].functions + parents[1].functions))
                    
                    # Add some random new functions
                    if random.random() < mutation_rate:
                        extra_funcs = ['sin', 'cos', 'wave', 'flow', 'turbulence']
                        child.functions.extend(random.sample(extra_funcs, 
                                                           k=random.randint(1, 3)))
                    
                    new_population.append(child)
                else:
                    # Create completely new individual
                    new_population.append(EnhancedArtisticFunction(self.intent))
            
            # Ensure some completely new individuals
            num_new = max(1, self.population_size // 10)
            new_population[-num_new:] = [EnhancedArtisticFunction(self.intent) 
                                       for _ in range(num_new)]
            
            population = new_population
            
            # Save best artwork
            best_idx = np.argmax(scores)
            best_artwork = artworks[best_idx]
            self.save_artwork(best_artwork, f"{output_dir}/generation_{generation}.png")
            print(f"Generation {generation}: Best Score = {scores[best_idx]:.3f}")
            print(f"Mutation Rate: {mutation_rate:.2f}")
    
    def save_artwork(self, artwork, filename):
        Image.fromarray((artwork * 255).astype(np.uint8)).save(filename)

if __name__ == "__main__":
    # Create and run the enhanced evolutionary art system
    art_system = EnhancedEvolutionaryArt(width=800, height=800)
    art_system.evolve()
    print("Evolution complete! Check the evolution_output directory for results.")