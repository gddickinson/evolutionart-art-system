"""
Evolutionary Art System package.

Modules:
    core        - ArtisticFunction, expression generation, mutation
    fitness     - Aesthetic evaluation and color harmony scoring
    rendering   - Image creation, layer composition, post-processing
    styles      - ArtisticIntent, moods, style-specific functions
    cli         - Command-line interface
"""

from evolutionary_art.core import ArtisticFunction, EvolutionaryArtSystem
from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction
from evolutionary_art.rendering import EnhancedEvolutionaryArt

__all__ = [
    "ArtisticFunction",
    "EvolutionaryArtSystem",
    "ArtisticIntent",
    "EnhancedArtisticFunction",
    "EnhancedEvolutionaryArt",
]
