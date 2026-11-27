"""Dataset generation, reference data and feature engineering."""

from .dataset import Dataset, ensure_dataset, generate_and_save, load_dataset
from .features import FEATURE_NAMES, FeatureBuilder, build_training_matrix
from .generator import generate_dataset

__all__ = [
    "Dataset",
    "FEATURE_NAMES",
    "FeatureBuilder",
    "build_training_matrix",
    "ensure_dataset",
    "generate_and_save",
    "generate_dataset",
    "load_dataset",
]
