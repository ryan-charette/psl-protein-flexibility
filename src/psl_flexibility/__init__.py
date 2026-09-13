"""Internal numerical helpers for the protein B-factor research study.

The legacy exports below support old fixtures; corrected operators live in
``psl_flexibility.sheaf``. There is no versioned public software API.
"""

from psl_flexibility.features import FeatureConfig, feature_names, features_for_coordinates, features_for_residues
from psl_flexibility.native_psl import NativePersistentSheafLaplacian

__version__ = "0.1.0"

__all__ = [
    "FeatureConfig",
    "NativePersistentSheafLaplacian",
    "__version__",
    "feature_names",
    "features_for_coordinates",
    "features_for_residues",
]
