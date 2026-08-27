"""Financial Agent — predicts trades for a given stock using ML on price/volume features."""

from .agent import FinancialAgent
from .config import Config

__all__ = ["FinancialAgent", "Config"]
__version__ = "1.0.0"
