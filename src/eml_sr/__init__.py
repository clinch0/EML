"""eml-sr: symbolic regression on EML trees, eml(x, y) = exp(x) - ln(y)."""

from eml_sr.api import EMLRegressor

__version__ = "0.1.0"
__all__ = ["EMLRegressor", "__version__"]
