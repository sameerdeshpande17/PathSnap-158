"""PyTorch MLP candidate-ranking model with CUDA support and checkpointing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pathsnap.models.base import ModelWrapper, Predictions


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32),
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.n_features = n_features
        self.hidden = hidden
        self.dropout = dropout

    def forward(self, x):
        return self.net(x).squeeze(1)

    def architecture_dict(self) -> dict[str, Any]:
        return dict(n_features=self.n_features, hidden=list(self.hidden),
                    dropout=self.dropout)


class MLPModel(ModelWrapper):
    device: str
    supports_resume_flag = True

    def __init__(self, name: str = "mlp", params: dict | None = None,
                 n_features: int | None = None,
                 feature_names: list[str] | None = None,
                 random_state: int = 42, device: str | None = None,
                 force_cpu: bool = False):
        self.name = name
        self.params = dict(params or {})
        self.random_state = random_state
        self.feature_names = list(feature_names or [])
        self.n_features = int(n_features or 0)
        self._fitted = False
        self.metrics: dict[str, Any] = {}
        self.classes_ = np.array([0, 1])
        self.model: _MLP | None = None
        self.optimizer_state: dict | None = None
        self.scheduler_state: dict | None = None
        self.early_stopping_state: dict | None = None
        self.rng_states: dict | None = None
        self.epochs_completed = 0

        # Device selection: never silently change the selected device.
        requested = self.params.get("device", "cuda" if torch.cuda.is_available()
                                    else "cpu")
        if force_cpu:
            device = "cpu"
        elif device is not None:
            device = device
        else:
            device = requested
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Set device='cpu' explicitly or check the CUDA installation.")
        self.device = device
        torch.manual_seed(random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed(random_state)

    def _build_model(self) -> _MLP:
        hidden = tuple(self.params.get("hidden", [64, 32]))
        dropout = float(self.params.get("dropout", 0.1))
        return _MLP(self.n_features, hidden=hidden, dropout=dropout).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.shape[1] != self.n_features and self.n_features:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features}, got {X.shape[1]}")
        params = self.params
        epochs = int(params.get("epochs", 15))
        batch_size = int(params.get("batch_size", 4096))
        lr = float(params.get("learning_rate", 0.001))
        weight_decay = float(params.get("weight_decay", 0.0001))
        patience = int(params.get("early_stopping_patience", 5))
        min_delta = float(params.get("early_stopping_min_delta", 1e-4))

        t0 = time.perf_counter()
        self.model = self._build_model()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 2))

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        n = len(X_t)
        if X_val is not None:
            Xv_t = torch.tensor(np.asarray(X_val, dtype=np.float32),
                                device=self.device)
            yv_t = torch.tensor(np.asarray(y_val, dtype=np.float32),
                                device=self.device)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        log_entries = []
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            running = 0.0
            for start_i in range(0, n, batch_size):
                idx = perm[start_i:start_i + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * len(idx)
            train_loss = running / n
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(Xv_t)
                    val_loss = float(criterion(val_logits, yv_t))
                self.model.train()
                scheduler.step(val_loss)
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    log_entries.append(dict(
                        epoch=epoch + 1, train_loss=train_loss,
                        val_loss=val_loss, early_stopped=True))
                    break
            log_entries.append(dict(epoch=epoch + 1, train_loss=train_loss,
                                    val_loss=val_loss))
            print(f"  [{self.name}] epoch {epoch + 1}/{epochs} "
                  f"train_loss={train_loss:.6f} "
                  f"{'val_loss=%.6f' % val_loss if val_loss is not None else ''}",
                  flush=True)

        self.model.eval()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        # Save states for resume
        self.optimizer_state = {}
        for group in optimizer.state_dict().get("param_groups", []):
            self.optimizer_state.setdefault("param_groups", []).append(
                {k: v for k, v in group.items() if k != "params"})
        for k, v in optimizer.state_dict().get("state", {}).items():
            self.optimizer_state.setdefault("state", {})[str(k)] = {
                kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                for kk, vv in v.items()
            }
        self.scheduler_state = scheduler.state_dict()
        self.early_stopping_state = dict(best_val=best_val, no_improve=no_improve,
                                         patience=patience, min_delta=min_delta)
        self.rng_states = dict(
            torch_rng=torch.get_rng_state().numpy().tobytes(),
            cuda_rng=b"".join(t.numpy().tobytes()
                              for t in torch.cuda.get_rng_state_all())
            if self.device == "cuda" else b"",
            numpy_rng=np.random.get_state()[1].tobytes(),
        )
        self.epochs_completed = len(log_entries)
        train_s = time.perf_counter() - t0
        self._fitted = True
        self.classes_ = np.array([0, 1])
        self.metrics = {
            "train_seconds": train_s,
            "device": self.device,
            "n_train_rows": int(n),
            "n_features": int(X.shape[1]),
            "epochs_completed": self.epochs_completed,
            "best_val_loss": best_val if best_state is not None else None,
            "training_log": log_entries,
        }
        return dict(self.metrics)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32),
                           device=self.device)
        batch = int(self.params.get("inference_batch_size", 8192))
        with torch.no_grad():
            probs = torch.cat([
                torch.sigmoid(self.model(X_t[i:i + batch]))
                for i in range(0, len(X_t), batch)
            ])
        return probs.detach().cpu().numpy()

    def save(self, directory: Path | str) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.pt"
        torch.save({
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in self.model.state_dict().items()},
            "architecture": self.model.architecture_dict(),
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "early_stopping_state": self.early_stopping_state,
            "rng_states": self.rng_states,
            "epochs_completed": self.epochs_completed,
        }, model_path)
        meta = dict(
            model_name=self.name, device=self.device,
            feature_names=self.feature_names, n_features=self.n_features,
            params=self.params, random_state=self.random_state,
            metrics=self.metrics, classes_=self.classes_.tolist(),
            requires_route=False,
        )
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
        return dict(path=str(model_path), bytes=model_path.stat().st_size,
                    sha256=_sha(model_path))

    @classmethod
    def load(cls, directory: Path | str) -> "MLPModel":
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        obj = cls(name=meta.get("model_name", "mlp"),
                  params=meta.get("params", {}),
                  n_features=meta.get("n_features", 0),
                  feature_names=meta.get("feature_names", []),
                  random_state=meta.get("random_state", 42),
                  device=meta.get("device"),
                  force_cpu=False)
        checkpoint = torch.load(directory / "model.pt",
                                map_location="cpu", weights_only=False)
        arch = checkpoint["architecture"]
        obj.model = _MLP(
            n_features=arch["n_features"],
            hidden=tuple(arch["hidden"]),
            dropout=arch.get("dropout", 0.1),
        ).to(obj.device)
        obj.model.load_state_dict(checkpoint["state_dict"])
        obj.model.eval()
        obj.optimizer_state = checkpoint.get("optimizer_state")
        obj.scheduler_state = checkpoint.get("scheduler_state")
        obj.early_stopping_state = checkpoint.get("early_stopping_state")
        obj.rng_states = checkpoint.get("rng_states")
        obj.epochs_completed = checkpoint.get("epochs_completed", 0)
        obj._fitted = True
        obj.classes_ = np.array(meta.get("classes_", [0, 1]))
        obj.metrics = meta.get("metrics", {})
        return obj

    def supports_resume(self) -> bool:
        return True

    def requires_route(self) -> bool:
        return False


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
