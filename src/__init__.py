"""
India Capacity Expansion Model
==============================
A least-cost optimization model for power sector planning.

Modules:
- data_loader: Functions to load and preprocess input data
- model: Core optimization model using PuLP
- visualizations: Plotting functions for results
"""

from .data_loader import DataLoader
from .model import CapacityExpansionModel
from .visualizations import ResultsVisualizer

__version__ = "1.0.0"
__author__ = "Your Name"
