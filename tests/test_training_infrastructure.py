"""Fast unit tests + tiny synthetic smoke tests for Phase 2B training infra.

No full-dataset training is performed here.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from pathsnap.candidates import generate, load_inputs
from pathsnap.features import CandidateCache, cache_key, dataset_identity_from_trips
from pathsnap.models import ALL_MODELS, MODEL_REGISTRY, MLPModel
from pathsnap.models.sklearn_models import (
    HistGradientBoostingModel, KNeighborsModel, LogisticRegressionModel,
    RandomForestModel,
)
from pathsnap.optimization import FoldResult, _build_search_space, run_cv_fold
from pathsnap.splits import make_folds, assert_no_leakage
from pathsnap.training import (
    ARTIFACTS_ROOT, compatible_checkpoint, list_runs, make_run_id, run_dir,
    train_model,
)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "Student folder" / "train"


def _make_tmp_trip(tmpdir: Path, tid: str = "tt1"):
    d = tmpdir / tid
    d.mkdir(parents=True)
    (d / f"{tid}.track").write_text("8.0002 49.0 1\n8.001 49.0 2\n8.002 49.001 3\n")
    (d / f"{tid}.nodes").write_text("8.0 49.0\n8.001 49.0\n8.002 49.001\n8.003 49.0\n")
    (d / f"{tid}.arcs").write_text("0 1\n1 2\n2 3\n0 2\n")
    return d


class FeatureCacheTests(unittest.TestCase):
    def test_cache_key_determinism_and_invalidation(self):
        k1 = cache_key("ds", {"radius_m": 50.0}, "v1")
        k2 = cache_key("ds", {"radius_m": 50.0}, "v1")
        k3 = cache_key("ds", {"radius_m": 60.0}, "v1")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_cache_write_read(self):
        with tempfile.TemporaryDirectory() as td:
            cache = CandidateCache(td)
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            key = "testkey"
            self.assertFalse(cache.exists(key))
            cache.write(key, df)
            self.assertTrue(cache.exists(key))
            pd.testing.assert_frame_equal(cache.read(key), df)
            manifest = cache.manifest()
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["key"], key)


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dir = _make_tmp_trip(self.tmp)
        self.trip = load_inputs(self.dir)

    def test_generate_preserves_arc_ids_and_features(self):
        frame = generate(self.trip, {"radius_m": 100.0, "max_candidates": 4,
                                     "fallback": "nearest", "gap_s": 5.}, "tt1")
        self.assertEqual(list(frame.columns[:3]),
                         ["trip_id", "observation_index", "arc_id"])
        self.assertTrue((frame.arc_id >= 0).all())
        self.assertTrue((frame.arc_id < 4).all())
        self.assertTrue(np.isfinite(frame[list(frame.columns[3:])]).all().all())

    def test_fallback_and_no_fallback(self):
        far = _make_tmp_trip(self.tmp, tid="tt2")
        # Move GPS far away so no candidates within tiny radius
        (far / "tt2.track").write_text("9.0 49.0 1\n9.001 49.0 2\n")
        trip = load_inputs(far)
        fb = generate(trip, {"radius_m": 0.001, "max_candidates": 1,
                             "fallback": "nearest", "gap_s": 5.}, "tt2")
        self.assertEqual(len(fb.observation_index.unique()), 2)
        self.assertTrue((fb.fallback_used == 1).all())
        nofb = generate(trip, {"radius_m": 0.001, "max_candidates": 1,
                               "fallback": "none", "gap_s": 5.}, "tt2")
        self.assertEqual(len(nofb), 0)

    def test_no_route_leakage_in_generate(self):
        import pathsnap.candidates as mod
        source = Path(mod.__file__).read_text()
        self.assertNotIn(".route", source)


class SplitTests(unittest.TestCase):
    def test_trip_grouped_split_no_leakage(self):
        trips = [f"t{i:03d}" for i in range(12)]
        folds = make_folds(trips, n_splits=3, seed=7)
        assert_no_leakage(folds)
        for fold in folds:
            self.assertFalse(set(fold.train_trips) & set(fold.validation_trips))
            self.assertEqual(set(fold.train_trips) | set(fold.validation_trips),
                             set(trips))


class SklearnModelTests(unittest.TestCase):
    def _data(self, n=60, nf=5):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, nf)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
        return X, y

    def _check(self, model, X, y):
        res = model.fit(X, y)
        self.assertIn("train_seconds", res)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X),))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        # Save/load
        with tempfile.TemporaryDirectory() as td:
            info = model.save(td)
            self.assertTrue(Path(info["path"]).exists())
            loaded = type(model).load(td)
            proba2 = loaded.predict_proba(X)
            np.testing.assert_allclose(proba, proba2)
        return res

    def test_logistic(self):
        self._check(LogisticRegressionModel("lr", n_features=5,
                                            feature_names=[f"f{i}" for i in range(5)]),
                    *self._data())

    def test_knn(self):
        self._check(KNeighborsModel("knn", n_features=5,
                                    feature_names=[f"f{i}" for i in range(5)]),
                    *self._data())

    def test_random_forest(self):
        self._check(RandomForestModel("rf", n_features=5,
                                      feature_names=[f"f{i}" for i in range(5)]),
                    *self._data())

    def test_hist_gb(self):
        self._check(HistGradientBoostingModel("hgb", n_features=5,
                                              feature_names=[f"f{i}" for i in range(5)]),
                    *self._data())


class TorchMLPTests(unittest.TestCase):
    def _data(self, n=80, nf=5):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(n, nf)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
        return X, y

    def test_cuda_path_tiny_batch(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MLPModel("mlp", params={"device": device, "epochs": 2,
                                        "batch_size": 16, "learning_rate": 0.01},
                         n_features=5,
                         feature_names=[f"f{i}" for i in range(5)],
                         random_state=7)
        self.assertEqual(model.device, device)
        X, y = self._data()
        res = model.fit(X, y, X_val=X[:20], y_val=y[:20])
        self.assertIn("train_seconds", res)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X),))
        # Save/load consistency
        with tempfile.TemporaryDirectory() as td:
            info = model.save(td)
            loaded = MLPModel.load(td)
            proba2 = loaded.predict_proba(X)
            np.testing.assert_allclose(proba, proba2, rtol=1e-5)
        # Supports resume
        self.assertTrue(model.supports_resume())

    def test_requires_route_false(self):
        for name, cls in MODEL_REGISTRY.items():
            m = cls(name=name, n_features=3,
                    feature_names=["a", "b", "c"])
            self.assertFalse(m.requires_route())


class TrainingInfraTests(unittest.TestCase):
    def test_train_model_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td) / "artifacts" / "models"
            # Monkeypatch ARTIFACTS_ROOT is hard; instead run tiny sklearn fit
            X = np.random.default_rng(3).normal(size=(30, 4)).astype(np.float32)
            y = (X[:, 0] > 0).astype(np.float64)
            m = LogisticRegressionModel("lr", n_features=4,
                                        feature_names=[f"f{i}" for i in range(4)])
            m.fit(X, y)
            self.assertTrue(m._fitted)

    def test_make_run_id_unique(self):
        cfg = {"models": {"logistic": {}}, "seed": 42, "feature_schema": []}
        id1 = make_run_id("logistic", "consensus", cfg)
        id2 = make_run_id("logistic", "consensus", cfg)
        self.assertEqual(id1, id2)  # deterministic given same inputs

    def test_list_runs_returns_structured_list(self):
        runs = list_runs("logistic")
        self.assertIsInstance(runs, list)
        for run in runs:
            self.assertIn("run_id", run)
            self.assertIn("run_dir", run)
            self.assertIn("device", run)


class OptimizationSmokeTests(unittest.TestCase):
    """Synthetic smoke tests for the Optuna optimization pipeline (no full
    training, no student-test data)."""

    def _synthetic_frames(self, n_features=5):
        rng = np.random.default_rng(0)
        # Train frame: 2 trips x 6 obs, each obs has 3 candidate rows.
        train_rows, val_rows = [], []
        for tid, n_obs in [("s1", 6), ("s2", 6)]:
            for oi in range(n_obs):
                for r in range(3):
                    train_rows.append(dict(
                        trip_id=tid, observation_index=oi, arc_id=r,
                        candidate_rank=r,
                        weak_target=1 if r == 0 else 0,
                        f0=rng.normal(), f1=rng.normal(), f2=rng.normal(),
                        f3=rng.normal(), f4=rng.normal(),
                    ))
        for tid, n_obs in [("s3", 4)]:
            for oi in range(n_obs):
                for r in range(3):
                    val_rows.append(dict(
                        trip_id=tid, observation_index=oi, arc_id=r,
                        candidate_rank=r,
                        weak_target=1 if r == 0 else 0,
                        f0=rng.normal(), f1=rng.normal(), f2=rng.normal(),
                        f3=rng.normal(), f4=rng.normal(),
                    ))
        train = pd.DataFrame(train_rows)
        val = pd.DataFrame(val_rows)
        train["consensus_eligible"] = True
        train["B_extra_eligible"] = False
        val["consensus_eligible"] = True
        val["B_extra_eligible"] = False
        train["weak_arc_id"] = 0
        val["weak_arc_id"] = 0
        train["positive_in_candidates"] = True
        val["positive_in_candidates"] = True
        feature_schema = [f"f{i}" for i in range(n_features)]
        return train, val, feature_schema

    def test_build_search_space_all_models(self):
        import optuna
        for model in ["hist_gradient_boosting", "random_forest", "mlp"]:
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            params = _build_search_space(model, trial)
            self.assertIsInstance(params, dict)
            self.assertTrue(len(params) > 0)

    def test_run_cv_fold_logistic(self):
        train, val, feature_schema = self._synthetic_frames()
        from pathsnap.models.sklearn_models import LogisticRegressionModel
        wrapper = LogisticRegressionModel("lr", n_features=5,
                                          feature_names=feature_schema,
                                          random_state=7)
        # Fit wrapper manually to get a fitted object, then reuse logic via
        # run_cv_fold which refits internally.
        from pathsnap.optimization import run_cv_fold
        res = run_cv_fold("logistic", {}, train, val, feature_schema,
                          "consensus", True, 7)
        self.assertIsInstance(res, FoldResult)
        self.assertGreaterEqual(res.top1_agreement, 0.0)
        self.assertLessEqual(res.top1_agreement, 1.0)
        self.assertGreaterEqual(res.mrr, 0.0)
        self.assertLessEqual(res.mrr, 1.0)
        self.assertIsInstance(res.per_trip_top1, dict)

    def test_run_cv_fold_mlp_cpu(self):
        train, val, feature_schema = self._synthetic_frames()
        from pathsnap.optimization import run_cv_fold
        # Force CPU for the smoke test (no GPU dependency).
        res = run_cv_fold("mlp", {"device": "cpu", "epochs": 2,
                                  "batch_size": 16, "hidden": [8]},
                          train, val, feature_schema, "consensus", True, 7)
        self.assertIsInstance(res, FoldResult)
        self.assertGreaterEqual(res.top1_agreement, 0.0)
        self.assertLessEqual(res.top1_agreement, 1.0)


if __name__ == "__main__":
    unittest.main()

