import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from pathsnap.alignment import Geometry, geometry, solve, diagnostics, transition_cost
from pathsnap.alignment_breaks import directed_graph, connection
from pathsnap.alignment_run import RAW, training_paths
from pathsnap.data import Trip, read_table


CONFIG = dict(sigma_m=5., beta_m=10., max_speed_mps=20., speed_slack_m=30.,
              progress_weight=1., speed_penalty=4., break_penalty=8.,
              beam_width=20, candidate_count=20, residual_limit_m=30.,
              ambiguity_m=2., gap_s=5.)


def lattice(distances, progress=None, dt=None, breaks=None):
    distances = np.asarray(distances, dtype=float)
    n, m = distances.shape
    if progress is None:
        progress = np.tile(np.arange(m) * 10., (n, 1))
    return Geometry(distances, np.asarray(progress, float), np.arange(m) * 10.,
                    np.ones(m) * 10., np.zeros(m, int) if breaks is None else np.array(breaks),
                    np.r_[0., np.full(n - 1, 5.)],
                    np.r_[0., np.ones(n - 1)] if dt is None else np.array(dt),
                    np.zeros((n, 2)), np.zeros((m + 1, 2)))


class AlignmentTests(unittest.TestCase):
    def test_directed_geometry_and_repeated_ids_are_occurrences(self):
        trip = Trip(np.array([[8.0002, 49., 1.], [8.0008, 49., 2.]]),
                    np.array([[8., 49.], [8.001, 49.]]),
                    np.array([[0, 1], [1, 0]]), np.array([0, 1, 0]))
        g = geometry(trip)
        self.assertEqual(g.distances.shape, (2, 3))
        np.testing.assert_allclose(g.distances[:, 0], g.distances[:, 2])
        self.assertGreater(g.progress[1, 0], g.progress[0, 0])
        self.assertLess(g.progress[1, 1], g.progress[0, 1])
        self.assertGreater(g.progress[0, 2], g.progress[0, 0])

    def test_multiple_points_can_dwell_on_one_occurrence(self):
        g = lattice([[0, 100], [1, 100], [2, 100], [1, 100]])
        for method in ["A", "B", "C"]:
            path, _ = solve(g, method, CONFIG)
            np.testing.assert_array_equal(path, [0, 0, 0, 0])

    def test_revisit_is_not_collapsed(self):
        # Occurrences 0 and 4 can share the same arc geometry, but chronology
        # requires visiting occurrence 4 after intervening route positions.
        g = lattice([[0, 50, 50, 50, 0], [50, 0, 50, 50, 50],
                     [50, 50, 0, 50, 50], [50, 50, 50, 0, 50],
                     [0, 50, 50, 50, 0]])
        for method in ["B", "C"]:
            path, _ = solve(g, method, CONFIG)
            np.testing.assert_array_equal(path, [0, 1, 2, 3, 4])

    def test_monotone_even_when_nearest_moves_backwards(self):
        g = lattice([[1, 0, 4], [0, 1, 4], [4, 1, 0]])
        path, _ = solve(g, "A", CONFIG)
        self.assertTrue((np.diff(path) < 0).any())
        for method in ["B", "C"]:
            path, _ = solve(g, method, CONFIG)
            self.assertTrue((np.diff(path) >= 0).all())

    def test_empty_fractional_and_out_of_bounds_routes_rejected(self):
        track = np.array([[8., 49., 1.]])
        for route in [np.array([], int), np.array([.5]), np.array([2]), np.array([-1])]:
            with self.assertRaises(ValueError):
                geometry(Trip(track, np.array([[8., 49.], [8.1, 49.]]),
                              np.array([[0, 1]]), route))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.route"
            p.write_text("0 1\n")
            with self.assertRaises(ValueError):
                read_table(p, 1, True)

    def test_directed_graph_disconnection_and_duplicate_edges(self):
        nodes = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.]])
        graph = directed_graph(nodes, np.array([[0, 1], [0, 1], [2, 3]]))
        distance, path = connection(graph, 0, 1)
        self.assertEqual(distance, 10)
        self.assertEqual(path, [0, 1])
        self.assertIsNone(connection(graph, 1, 0)[0])
        self.assertIsNone(connection(graph, 1, 2)[0])

    def test_large_gap_relaxes_speed_not_occurrence_order(self):
        progress = [[0, 1000], [0, 1000]]
        short = lattice([[0, 100], [100, 0]], progress, dt=[0, 1])
        long = lattice([[0, 100], [100, 0]], progress, dt=[0, 100])
        self.assertTrue(np.isinf(transition_cost(short, 1, np.array([0]),
                                                np.array([1]), "C", CONFIG)[0, 0]))
        self.assertTrue(np.isfinite(transition_cost(long, 1, np.array([0]),
                                                   np.array([1]), "C", CONFIG)[0, 0]))
        path, _ = solve(long, "C", CONFIG)
        np.testing.assert_array_equal(path, [0, 1])
        _, _, flags, eligible = diagnostics(long, path, CONFIG)
        self.assertTrue(flags["large_gap_adjacent"].all())
        self.assertFalse(eligible.any())

    def test_broken_boundary_is_flagged_not_repaired(self):
        g = lattice([[0, 100], [100, 0]], breaks=[0, 1])
        for method in ["B", "C"]:
            path, _ = solve(g, method, CONFIG)
            np.testing.assert_array_equal(path, [0, 1])
            self.assertTrue(diagnostics(g, path, CONFIG)[2]["reference_break_crossing"][1])

    def test_full_state_dp_against_independent_exhaustive_oracle(self):
        rng = np.random.default_rng(42)
        for method in ["B", "C"]:
            for _ in range(6):
                distances = rng.uniform(0, 20, (4, 3))
                g = lattice(distances)
                path, details = solve(g, method, CONFIG)
                costs = []
                for sequence in itertools.combinations_with_replacement(range(3), 4):
                    energy = (distances[0, sequence[0]] / 5 if method == "B" else
                              .5 * (distances[0, sequence[0]] / 5) ** 2)
                    for i in range(1, 4):
                        prev, curr = sequence[i - 1:i + 1]
                        step_costs = np.array([abs((j - prev) * 10 - 5) / 10
                                               if j >= prev else np.inf for j in range(3)])
                        if method == "C":
                            step_costs += logsumexp(-step_costs)
                        energy += step_costs[curr]
                        energy += (distances[i, curr] / 5 if method == "B" else
                                   .5 * (distances[i, curr] / 5) ** 2)
                    costs.append((energy, sequence))
                expected, sequence = min(costs)
                self.assertAlmostEqual(details["objective"], expected, places=9)
                np.testing.assert_array_equal(path, sequence)

    def test_beam_retention_and_extreme_residuals_are_finite(self):
        g = lattice([[0, 100000], [100000, 0], [0, 100000]],
                    progress=[[0, 10000], [500, 10000], [0, 10000]])
        for method in ["B", "C"]:
            path, detail = solve(g, method, CONFIG | {"beam_width": 1, "candidate_count": 1})
            self.assertTrue(np.isfinite(detail["objective"]))
            self.assertTrue((np.diff(path) >= 0).all())

    def test_training_partition_guard(self):
        self.assertEqual(len(training_paths([RAW / "train/00000001"])), 1)
        for paths in [[], [RAW / "student_test/00000000"],
                      [RAW / "train/00000001", RAW / "student_test/00000000"]]:
            with self.assertRaises(ValueError):
                training_paths(paths)

    def test_projection_scale_and_timestamp_validation(self):
        trip = Trip(np.array([[8., 49., 1.], [8.001, 49., 2.]]),
                    np.array([[8., 49.], [8.001, 49.]]),
                    np.array([[0, 1]]), np.array([0]))
        g = geometry(trip)
        self.assertTrue(70 < g.lengths[0] < 75)
        trip.track[1, 2] = 1
        with self.assertRaises(ValueError):
            geometry(trip)


if __name__ == "__main__":
    unittest.main()
