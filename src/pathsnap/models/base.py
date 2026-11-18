"""Abstract interface for every candidate-ranking model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Predictions:
    scores: np.ndarray      # shape (n_candidates,) probability of positive
    arc_ids: np.ndarray     # shape (n_candidates,) directed arc id
    observation_index: np.ndarray


class ModelWrapper:
    """Every model exposes fit/predict/save/load plus metadata."""

    name: str
    device: str
    feature_names: list[str]
    n_features: int
    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path | str) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path | str) -> "ModelWrapper":
        raise NotImplementedError

    def requires_route(self) -> bool:
        """True if the model needs .route at inference. Must be False."""
        return False

    def supports_resume(self) -> bool:
        return False
