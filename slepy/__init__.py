"""
slepy: Python library for calculating sea level contribution from ice sheet model output

This library provides tools to calculate sea level contribution from ice sheet 
thickness and bed elevation data using the methodology from Goelzer et al. (2020).
"""

__version__ = "1.0.0"
__author__ = "Jonnie Barnsley"

from .core import SLECalculator
from .utils import scale_factor
from .config import DENSITIES, OCEAN_AREA

__all__ = [
    "SLECalculator",
    "scale_factor",
    "DENSITIES",
    "OCEAN_AREA",
]