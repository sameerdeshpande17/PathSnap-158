"""Phase 2B verification: candidates, features, models, splits, no route leakage."""
import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import (
    FEATURES, SCHEMA_VERSION, InferenceTrip, generate, load_inputs,
    validate_inputs,
)
from pathsnap.explore import ROOT

# Project root computed directly (avoids importing the heavy explore module).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_trip(tmp, track, nodes, arcs, tid="t1"):
    d = Path(tmp) / tid
    d.mkdir()
    prefix = d / tid
    prefix.with_suffix(".track").write_text(
        "\n".join(f"{a} {b} {c}" for a, b, c in track))
    prefix.with_suffix(".nodes").write_text(
        "\n".join(f"{a} {b}" for a, b in nodes))
    prefix.with_suffix(".arcs").write_text(
        "\n".join(f"{a} {b}" for a, b in arcs))
    return d


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = make_trip(
            self.tmp,
            track=[(8.0002, 49.0, 1.), (8.001, 49.0, 2.), (8.002, 49.001, 3.)],
            nodes=[(8., 49.), (8.001, 49.), (8.002, 49.001), (8.003, 49.)],
            arcs=[(0, 1), (1, 2), (2, 3), (0, 2)],
            tid="t1",
        )
        self.trip = load_inputs(self.dir)

    def test_columns_ids_and_bounds_preserved(self):
        frame = generate(self.trip, {"radius_m": 100., "max_candidates": 5,
                                     "fallback": "nearest", "gap_s": 5.}, "t1")
        self.assertEqual(list(frame.columns[:3]),
                         ["trip_id", "observation_index", "arc_id"])
        self.assertTrue((frame.trip_id == "t1").all())
        self.assertTrue((frame.arc_id >= 0).all())
        self.assertTrue((frame.arc_id < 4).all())
        self.assertTrue((frame.observation_index < 3).all())

    def test_feature_schema_and_finiteness(self):
        frame = generate(self.trip, {"radius_m": 100., "max_candidates": 5,
                                     "fallback": "nearest", "gap_s": 5.}, "t1")
        self.assertEqual(list(frame.columns[:6]),
                         ["trip_id", "observation_index", "arc_id",
                          "longitude", "latitude", "timestamp"])
        self.assertEqual(list(frame.columns[6:]), FEATURES)
        self.assertTrue(np.isfinite(frame[FEATURES].to_numpy()).all())

    def test_fallback_covers_isolated_observations(self):
        far = make_trip(self.tmp, [(9.0, 49.0, 1.), (9.001, 49.0, 2.)],
                        [(8., 49.), (8.001, 49.)], [(0, 1)], tid="t3")
        trip = load_inputs(far)
        frame = generate(trip, {"radius_m": 0.001, "max_candidates": 1,
                                "fallback": "nearest", "gap_s": 5.}, "t3")
        self.assertEqual(len(frame.observation_index.unique()), 2)
        self.assertTrue((frame.fallback_used == 1).all())

    def test_no_fallback_omits_isolated_observations(self):
        far = make_trip(self.tmp, [(9.0, 49.0, 1.), (9.001, 49.0, 2.)],
                        [(8., 49.), (8.001, 49.)], [(0, 1)], tid="t4")
        trip = load_inputs(far)
        frame = generate(trip, {"radius_m": 0.001, "max_candidates": 1,
                                "fallback": "none", "gap_s": 5.}, "t4")
        self.assertEqual(len(frame), 0)

    def test_duplicate_parallel_arcs_remain_separate(self):
        d = make_trip(self.tmp, [(8.0005, 49.0, 1.)],
                      [(8., 49.), (8.001, 49.)], [(0, 1), (0, 1)], tid="t2")
        trip = load_inputs(d)
        frame = generate(trip, {"radius_m": 100., "max_candidates": 10,
                                "fallback": "none", "gap_s": 5.}, "t2")
        self.assertEqual(sorted(frame.arc_id.unique()), [0, 1])

    def test_invalid_inputs_rejected(self):
        bad = InferenceTrip(np.array([[8., 49., 1.]]),
                            np.array([[8., 49.], [900., 49.]]), np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            validate_inputs(bad)
        bad2 = InferenceTrip(np.array([[8., 49., 1.]]),
                             np.array([[8., 49.], [8.001, 49.]]), np.array([[0, 2]]))
        with self.assertRaises(ValueError):
            validate_inputs(bad2)


class ProjectionAndHeadingTests(unittest.TestCase):
    def test_longitude_first_projection_scale(self):
        from pathsnap.candidates import project
        track = np.array([[8., 49., 1.], [8.01, 49., 2.]])
        nodes = np.array([[8., 49.], [8.01, 49.]])
        gps, _ = project(track, nodes)
        # One degree of longitude near latitude 49 is about 734 metres.
        self.assertTrue(700 < gps[1, 0] - gps[0, 0] < 770)

    def test_heading_features_safe_for_stationary_and_gap_rows(self):
        from pathsnap.candidates import movement_features
        track = np.array([[8., 49., 1.], [8., 49., 2.], [8., 49.001, 3.]])
        gps = project_stub(track[:, :2])
        dt, step, heading, speed, valid, time_valid = movement_features(track, gps)
        self.assertFalse(valid[0])  # no previous observation
        self.assertFalse(time_valid[0])
        self.assertTrue(np.all(np.isfinite(heading)))
        self.assertTrue(np.all(np.isfinite(speed)))


def project_stub(ll):
    from pyproj import CRS, Transformer
    crs = CRS.from_proj4("+proj=aeqd +lat_0=49 +lon_0=8 +datum=WGS84 +units=m")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return np.column_stack(tr.transform(ll[:, 0], ll[:, 1]))


class SplitTests(unittest.TestCase):
    def test_trip_grouped_split_keeps_trips_intact(self):
        from sklearn.model_selection import GroupKFold
        trips = [f"t{i:03d}" for i in range(10)]
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in gkf.split(trips, groups=trips):
            train_trips = {trips[i] for i in train_idx}
            test_trips = {trips[i] for i in test_idx}
            self.assertFalse(train_trips & test_trips)


class SerializationTests(unittest.TestCase):
    def test_feature_schema_round_trip(self):
        payload = {"features": FEATURES, "version": SCHEMA_VERSION}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "schema.json"
            p.write_text(json.dumps(payload))
            loaded = json.loads(p.read_text())
        self.assertEqual(loaded["features"], FEATURES)
        self.assertEqual(loaded["version"], SCHEMA_VERSION)


class NoRouteLeakageTests(unittest.TestCase):
    def test_candidate_generation_takes_only_network_inputs(self):
        import pathsnap.candidates as mod
        source = Path(mod.__file__).read_text()
        self.assertNotIn(".route", source)
        sig = inspect.signature(generate)
        self.assertEqual(list(sig.parameters), ["trip", "config", "trip_id"])


class MRRRegressionTests(unittest.TestCase):
    """Regression tests for the MRR / population fixes (Phase 3A audit)."""

    def _make_frame(self):
        # Three observations, each with 3 candidates ranked by score.
        rows = []
        # obs 0: positive at rank 1 -> RR = 1.0
        rows += [
            ("t1", 0, 100, 1, 0.9, 1),
            ("t1", 0, 101, 0, 0.8, 2),
            ("t1", 0, 102, 0, 0.7, 3),
        ]
        # obs 1: positive at rank 2 -> RR = 0.5
        rows += [
            ("t1", 1, 200, 0, 0.9, 1),
            ("t1", 1, 201, 1, 0.8, 2),
            ("t1", 1, 202, 0, 0.7, 3),
        ]
        # obs 2: positive absent (candidate miss) -> RR = 0.0
        rows += [
            ("t1", 2, 300, 0, 0.9, 1),
            ("t1", 2, 301, 0, 0.8, 2),
        ]
        df = pd.DataFrame(
            rows,
            columns=["trip_id", "observation_index", "arc_id", "weak_target",
                     "candidate_score", "candidate_rank"],
        )
        # weak_arc_id = the labeled reference arc per observation.
        # obs 0: label arc 100 (top-1 correct). obs 1: label arc 201 (top-1 wrong,
        #        because arc 200 scores higher). obs 2: label arc 999 (absent ->
        #        candidate miss and top-1 wrong).
        df["weak_arc_id"] = [100, 100, 100, 201, 201, 201, 999, 999]
        df["consensus_eligible"] = True
        df["B_extra_eligible"] = False
        # positive_in_candidates is constant within an observation (whether the
        # labeled arc appears among the candidates).
        df["positive_in_candidates"] = [True, True, True, True, True, True,
                                        False, False]
        return df

    def test_reciprocal_ranks_values(self):
        from pathsnap.phase2b_metrics import reciprocal_ranks
        df = self._make_frame()
        # Add consensus eligibility column expected by reciprocal_ranks? It only
        # needs weak_target, trip_id, observation_index, candidate_score, rank.
        rr = reciprocal_ranks(df)
        self.assertEqual(rr[("t1", 0)], 1.0)  # positive at rank 1
        self.assertEqual(rr[("t1", 1)], 0.5)  # positive at rank 2
        self.assertEqual(rr[("t1", 2)], 0.0)  # positive absent

    def test_evaluate_frame_mrr_is_reciprocal(self):
        from pathsnap.phase2b_metrics import evaluate_frame
        df = self._make_frame()
        obs, _ = evaluate_frame(df)
        # Mean of [1.0, 0.5, 0.0] = 0.5
        self.assertAlmostEqual(obs["mrr"], 0.5)
        # Top-1 agreement: obs0 correct, obs1 wrong (top=200 but pos=201), obs2 miss
        self.assertAlmostEqual(obs["top1_agreement"], 1 / 3)
        self.assertEqual(obs["candidate_miss_count"], 1)

    def test_mrr_not_rank_position(self):
        """Ensure MRR is reciprocal rank, not raw rank position (the bug)."""
        from pathsnap.phase2b_metrics import reciprocal_ranks
        df = self._make_frame()
        rr = reciprocal_ranks(df)
        # If it returned raw rank position: [1, 2, 0] -> mean 1.0.
        # Reciprocal: [1.0, 0.5, 0.0] -> mean 0.5.
        self.assertAlmostEqual(float(rr.mean()), 0.5)
        self.assertNotAlmostEqual(float(rr.mean()), 1.0)

    def test_population_consensus_vs_union_counts(self):
        import os
        from pathsnap.candidates import generate, load_inputs
        from pathsnap.labels import build_label_table, join_labels_with_candidates
        raw = PROJECT_ROOT / "data/raw/Student folder/student_test"
        frames = []
        for d in sorted(os.listdir(raw)):
            td = os.path.join(raw, d)
            if not os.path.isdir(td):
                continue
            trip = load_inputs(td)
            gen = generate(trip, {"radius_m": 50.0, "max_candidates": 12,
                                  "fallback": "nearest", "gap_s": 5.}, d)
            labels = build_label_table("student_test", d)
            frames.append(join_labels_with_candidates(gen, labels))
        frame = pd.concat(frames, ignore_index=True)
        frame = frame[frame.weak_target >= 0].copy()
        consensus_obs = frame[frame.consensus_eligible].groupby(
            ["trip_id", "observation_index"]).ngroups
        union_obs = frame.groupby(["trip_id", "observation_index"]).ngroups
        # Phase 2A screened consensus = 12,516; Phase 2B union = 12,862.
        self.assertEqual(consensus_obs, 12516)
        self.assertEqual(union_obs, 12862)


if __name__ == "__main__":
    unittest.main()
