"""
Command-line interface for the evolutionary art system.

Usage:
    python -m evolutionary_art --style geometric --mood calm --generations 50 --size 800
"""

import argparse
import sys

from evolutionary_art.core import EvolutionaryArtSystem
from evolutionary_art.rendering import EnhancedEvolutionaryArt
from evolutionary_art.styles import ArtisticIntent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate art through evolutionary algorithms."
    )
    parser.add_argument(
        "--mode", choices=["basic", "enhanced"], default="enhanced",
        help="Art generation mode (default: enhanced).",
    )
    parser.add_argument(
        "--style", choices=ArtisticIntent.STYLES, default=None,
        help="Artistic style (default: random).",
    )
    parser.add_argument(
        "--mood", choices=ArtisticIntent.MOODS, default=None,
        help="Artistic mood (default: random).",
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of generations (default: 50).",
    )
    parser.add_argument(
        "--size", type=int, default=800,
        help="Image width and height in pixels (default: 800).",
    )
    parser.add_argument(
        "--output-dir", default="evolution_output",
        help="Output directory (default: evolution_output).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args(argv)

    if args.mode == "basic":
        system = EvolutionaryArtSystem(
            width=args.size, height=args.size, seed=args.seed,
        )
        system.generations = args.generations
        system.evolve_art(output_dir=args.output_dir)
    else:
        system = EnhancedEvolutionaryArt(
            width=args.size, height=args.size,
            seed=args.seed, mood=args.mood, style=args.style,
        )
        system.generations = args.generations
        system.evolve(output_dir=args.output_dir)

    print("Evolution complete! Check the output directory for results.")


if __name__ == "__main__":
    main()
