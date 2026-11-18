"""scikit-learn candidate-ranking models (CPU)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.models.base import ModelWrapper, Predictions


class _SklearnModel(ModelWrapper):
    device = "CPU"
    supports_resume_flag = False

    def __init__(self, name: str, params: dict | None = None,
                 n_features: int | None = None,
                 feature_names: list[str] | None = None,
                 random_state: int = 42):
        self.name = name
        self.params = dict(params or {})
        self.random_state = random_state
        self.feature_names = list(feature_names or [])
        self.n_features = int(n_features or 0)
        self.scaler = StandardScaler()
        self.estimator: BaseEstimator | None = None
        self.classes_ = np.array([0, 1])
        self._fitted = False
        self.metrics: dict[str, Any] = {}

    def _build_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[1] != self.n_features and self.n_features:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features}, got {X.shape[1]}")
        t0 = time.perf_counter()
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.estimator = self._build_estimator()
        self.estimator.fit(Xs, y)
        train_s = time.perf_counter() - t0
        self._fitted = True
        self.classes_ = np.array([0, 1])
        self.metrics = {"train_seconds": train_s, "device": self.device,
                        "n_train_rows": int(len(X)),
                        "n_features": int(X.shape[1])}
        if X_val is not None:
            val_proba = self.predict_proba(X_val)
            self.metrics["validation_rows"] = int(len(X_val))
            self.metrics["validation_mean_score"] = float(np.mean(val_proba))
        return dict(self.metrics)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(Xs)[:, 1]
        # decision_function -> logistic
        df = self.estimator.decision_function(Xs)
        return 1.0 / (1.0 + np.exp(-df))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def save(self, directory: Path | str) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        bundle = {
            "estimator": self.estimator,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "classes_": self.classes_,
            "params": self.params,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "model_name": self.name,
            "device": self.device,
            "requires_route": False,
        }
        path = directory / "pipeline.joblib"
        joblib.dump(bundle, path, compress=3)
        return dict(path=str(path), bytes=path.stat().st_size,
                    sha256=_sha(path))

    @classmethod
    def load(cls, directory: Path | str) -> "ModelWrapper":
        directory = Path(directory)
        bundle = joblib.load(directory / "pipeline.joblib")
        obj = cls(name=bundle.get("model_name", cls.__name__),
                  params=bundle.get("params", {}),
                  n_features=bundle.get("n_features", 0),
                  feature_names=bundle.get("feature_names", []),
                  random_state=bundle.get("random_state", 42))
        obj.estimator = bundle["estimator"]
        obj.scaler = bundle["scaler"]
        obj.classes_ = bundle["classes_"]
        obj._fitted = True
        obj.metrics = bundle.get("metrics", {})
        return obj

    def supports_resume(self) -> bool:
        return self.supports_resume_flag

    def requires_route(self) -> bool:
        return False


class LogisticRegressionModel(_SklearnModel):
    def _build_estimator(self) -> BaseEstimator:
        return LogisticRegression(
            C=self.params.get("C", 1.0),
            max_iter=self.params.get("max_iter", 500),
            solver="lbfgs",
            random_state=self.random_state,
        )


class KNeighborsModel(_SklearnModel):
    def _build_estimator(self) -> BaseEstimator:
        return KNeighborsClassifier(
            n_neighbors=self.params.get("n_neighbors", 11),
            weights=self.params.get("weights", "distance"),
            algorithm=self.params.get("algorithm", "ball_tree"),
            leaf_size=self.params.get("leaf_size", 40),
            n_jobs=self.params.get("n_jobs", 4),
        )


class RandomForestModel(_SklearnModel):
    def _build_estimator(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 18),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            max_features=self.params.get("max_features", "sqrt"),
            n_jobs=self.params.get("n_jobs", 4),
            random_state=self.random_state,
        )


class HistGradientBoostingModel(_SklearnModel):
    def _build_estimator(self) -> BaseEstimator:
        return HistGradientBoostingClassifier(
            max_iter=self.params.get("max_iter", 150),
            learning_rate=self.params.get("learning_rate", 0.08),
            max_leaf_nodes=self.params.get("max_leaf_nodes", 31),
            l2_regularization=self.params.get("l2_regularization", 1.0),
            early_stopping=self.params.get("early_stopping", False),
            random_state=self.random_state,
        )


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
