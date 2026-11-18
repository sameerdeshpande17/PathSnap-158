"""Pathsnap ML models package."""
from .base import ModelWrapper, Predictions
from .sklearn_models import LogisticRegressionModel, KNeighborsModel, \
    RandomForestModel, HistGradientBoostingModel
from .torch_mlp import MLPModel

__all__ = [
    "ModelWrapper", "Predictions",
    "LogisticRegressionModel", "KNeighborsModel", "RandomForestModel",
    "HistGradientBoostingModel", "MLPModel",
]

MODEL_REGISTRY = {
    "logistic": LogisticRegressionModel,
    "knn": KNeighborsModel,
    "random_forest": RandomForestModel,
    "hist_gradient_boosting": HistGradientBoostingModel,
    "mlp": MLPModel,
}

ALL_MODELS = sorted(MODEL_REGISTRY)
