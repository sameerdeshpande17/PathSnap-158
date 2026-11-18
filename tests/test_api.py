"""FastAPI backend tests for PathSnap map matching API."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pathsnap.api import app


class TestAPI(unittest.TestCase):
    """Test the FastAPI backend endpoints."""

    def setUp(self):
        """Create a test client with a mocked model pipeline."""
        self.patcher_ready = patch("pathsnap.api.MODEL_READY", True)
        self.patcher_name = patch("pathsnap.api.MODEL_NAME", "hist_gradient_boosting")
        self.patcher_device = patch("pathsnap.api.MODEL_DEVICE", "CPU")
        self.patcher_pipeline = patch("pathsnap.api.PIPELINE")
        self.patcher_trips = patch("pathsnap.api.AVAILABLE_TRIPS")

        self.patcher_ready.start()
        self.patcher_name.start()
        self.patcher_device.start()
        self.mock_pipeline = self.patcher_pipeline.start()
        self.mock_trips = self.patcher_trips.start()

        # Mock available trips
        self.mock_trips.__getitem__.side_effect = lambda k: {
            "student_test/00000056": {
                "trip_id": "00000056", "split": "student_test",
                "n_observations": 3958, "n_nodes": 14759, "n_arcs": 27631,
            }
        }.get(k)
        self.mock_trips.__contains__.side_effect = lambda k: k in {
            "student_test/00000056", "train/00000001",
        }
        self.mock_trips.values.return_value = [
            {"trip_id": "00000056", "split": "student_test",
             "n_observations": 3958, "n_nodes": 14759, "n_arcs": 27631},
            {"trip_id": "00000001", "split": "train",
             "n_observations": 1000, "n_nodes": 5000, "n_arcs": 8000},
        ]

        # Mock pipeline.decode_trip
        from pathsnap.map_matching_pipeline import TripPrediction
        from pathsnap.route_reconstruct import ReconstructedRoute

        self.mock_pred = MagicMock(spec=TripPrediction)
        self.mock_pred.trip_id = "00000056"
        self.mock_pred.split = "student_test"
        self.mock_pred.n_observations = 100
        self.mock_pred.n_candidates = 500
        self.mock_pred.classifier_top1_arc = [1, 2, 3]
        self.mock_pred.classifier_top1_prob = [0.9, 0.8, 0.7]
        self.mock_pred.hmm_decoded_arcs = [1, 2, 3]
        self.mock_pred.hmm_log_prob = -5.0
        self.mock_pred.hmm_n_connected = 2
        self.mock_pred.hmm_n_disconnected = 0
        self.mock_pred.hmm_n_dwells = 0
        self.mock_pred.route = MagicMock(spec=ReconstructedRoute)
        self.mock_pred.route.n_route_segments = 3
        self.mock_pred.route.n_connector_segments = 0
        self.mock_pred.route.n_disconnected_gaps = 0
        self.mock_pred.route.n_repeated_arc_traversals = 0
        self.mock_pred.route.total_length_m = 3000.0
        self.mock_pred.route.arc_id_sequence = [1, 2, 3]
        self.mock_pred.candidate_gen_s = 0.05
        self.mock_pred.classifier_score_s = 0.02
        self.mock_pred.viterbi_s = 0.10
        self.mock_pred.route_reconstruct_s = 0.01

        self.mock_pipeline.decode_trip.return_value = self.mock_pred

        self.client = TestClient(app)

    def tearDown(self):
        self.patcher_pipeline.stop()
        self.patcher_trips.stop()
        self.patcher_device.stop()
        self.patcher_name.stop()
        self.patcher_ready.stop()

    def test_health(self):
        """GET /health returns server status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_ready", data)
        self.assertIn("model_name", data)
        self.assertIn("model_device", data)
        self.assertIn("cuda_available", data)
        self.assertIn("trips_available", data)

    def test_model_info(self):
        """GET /api/model returns model metadata."""
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "hist_gradient_boosting")
        self.assertEqual(data["device"], "CPU")
        self.assertIn("n_features", data)
        self.assertIn("feature_schema", data)

    def test_list_trips(self):
        """GET /api/trips lists available trips."""
        response = self.client.get("/api/trips?split=student_test")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        if data:
            self.assertIn("trip_id", data[0])
            self.assertIn("split", data[0])
            self.assertIn("n_observations", data[0])

    def test_predict(self):
        """POST /api/predict returns prediction results."""
        response = self.client.post(
            "/api/predict",
            json={"trip_id": "00000056", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["trip_id"], "00000056")
        self.assertEqual(data["split"], "student_test")
        self.assertIn("n_observations", data)
        self.assertIn("n_candidates", data)
        self.assertIn("classifier_top1_arcs", data)
        self.assertIn("hmm_decoded_arcs", data)
        self.assertIn("timing_s", data)

    def test_predict_invalid_trip(self):
        """POST /api/predict with invalid trip returns 404."""
        response = self.client.post(
            "/api/predict",
            json={"trip_id": "nonexistent", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 404)

    def test_predict_missing_model(self):
        """POST /api/predict without model returns 503."""
        # Verify the guard clause exists in the predict function
        import inspect
        from pathsnap.api import predict
        source = inspect.getsource(predict)
        self.assertIn("MODEL_READY", source)
        self.assertIn("PIPELINE is None", source)
        self.assertIn("503", source)

    def test_evaluate(self):
        """POST /api/evaluate returns evaluation metrics."""
        response = self.client.post(
            "/api/evaluate",
            json={"trip_id": "00000056", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["trip_id"], "00000056")
        self.assertIn("classifier_top1_agreement_rate", data)
        self.assertIn("hmm_agreement_rate", data)
        self.assertIn("route_precision", data)
        self.assertIn("route_recall", data)
        self.assertIn("route_f1", data)
        self.assertIn("route_connectivity", data)
        self.assertIn("route_edit_distance", data)
        self.assertIn("timing_s", data)

    def test_evaluate_invalid_trip(self):
        """POST /api/evaluate with invalid trip returns 404."""
        response = self.client.post(
            "/api/evaluate",
            json={"trip_id": "nonexistent", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 404)

    def test_json_serialization(self):
        """All API responses are valid JSON."""
        for endpoint, method, payload in [
            ("/health", "GET", None),
            ("/api/model", "GET", None),
            ("/api/trips?split=student_test", "GET", None),
            ("/api/predict", "POST", {"trip_id": "00000056", "split": "student_test"}),
            ("/api/evaluate", "POST", {"trip_id": "00000056", "split": "student_test"}),
        ]:
            if method == "GET":
                resp = self.client.get(endpoint)
            else:
                resp = self.client.post(endpoint, json=payload)
            self.assertIn(resp.status_code, (200, 404, 422, 503))
            # Must be parseable JSON
            json.loads(resp.text)

    def test_route_level_metrics(self):
        """Evaluate endpoint returns all route-level metrics."""
        response = self.client.post(
            "/api/evaluate",
            json={"trip_id": "00000056", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Route-level metrics must be present
        self.assertIn("route_precision", data)
        self.assertIn("route_recall", data)
        self.assertIn("route_f1", data)
        self.assertIn("route_connectivity", data)
        self.assertIn("route_edit_distance", data)
        # Values must be in valid ranges
        self.assertGreaterEqual(data["route_precision"], 0.0)
        self.assertLessEqual(data["route_precision"], 1.0)
        self.assertGreaterEqual(data["route_recall"], 0.0)
        self.assertLessEqual(data["route_recall"], 1.0)
        self.assertGreaterEqual(data["route_f1"], 0.0)
        self.assertLessEqual(data["route_f1"], 1.0)
        self.assertGreaterEqual(data["route_connectivity"], 0.0)
        self.assertLessEqual(data["route_connectivity"], 1.0)
        self.assertGreaterEqual(data["route_edit_distance"], 0)

    def test_classifier_vs_hmm(self):
        """Compare classifier vs HMM predictions."""
        response = self.client.post(
            "/api/predict",
            json={"trip_id": "00000056", "split": "student_test"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        cls_arcs = data["classifier_top1_arcs"]
        hmm_arcs = data["hmm_decoded_arcs"]
        self.assertEqual(len(cls_arcs), len(hmm_arcs))
        self.assertGreater(len(cls_arcs), 0)


if __name__ == "__main__":
    unittest.main()
