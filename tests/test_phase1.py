import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

from pathsnap.baseline import adjacency_rate, nearest_road
from pathsnap.data import load_trip, project, read_table


class ParsingTests(unittest.TestCase):
    def test_rejects_ragged_nonfinite_and_fractional_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            for text, cols, integer in [
                ("1 2\n3\n", 2, False), ("nan 2\n", 2, False),
                ("1.5 2\n", 2, True), ("", 2, False),
            ]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_table(path, cols, integer)

    def test_whitespace_and_single_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_text("  0\t  2 \n")
            np.testing.assert_array_equal(read_table(path, 2, True), [[0, 2]])

    def test_arc_and_route_id_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "trip"
            directory.mkdir()
            (directory / "trip.track").write_text("8 49 1\n8.001 49 2\n")
            (directory / "trip.nodes").write_text("8 49\n8.001 49\n")
            (directory / "trip.arcs").write_text("0 2\n")
            (directory / "trip.route").write_text("0\n")
            with self.assertRaisesRegex(ValueError, "node ID"):
                load_trip(directory)
            (directory / "trip.arcs").write_text("0 1\n")
            (directory / "trip.route").write_text("1\n")
            with self.assertRaisesRegex(ValueError, "route arc ID"):
                load_trip(directory)

    def test_projection_order_and_metres(self):
        track = np.array([[8., 49., 1.], [8.001, 49., 2.]])
        gps, _ = project(track, track[:, :2])
        self.assertGreater(gps[1, 0] - gps[0, 0], 70)
        self.assertLess(gps[1, 0] - gps[0, 0], 75)
        self.assertLess(abs(gps[1, 1] - gps[0, 1]), .01)


class BaselineTests(unittest.TestCase):
    def test_segment_interior_and_deterministic_direction_tie(self):
        nodes = np.array([[0., 0.], [10., 0.], [0., 5.], [10., 5.]])
        arcs = np.array([[0, 1], [1, 0], [2, 3]])
        ids, distances, ties, snapped = nearest_road(
            np.array([[5., 1.], [15., 5.]]), nodes, arcs)
        np.testing.assert_array_equal(ids, [0, 2])
        np.testing.assert_allclose(distances, [1, 5])
        np.testing.assert_array_equal(ties, [2, 1])
        np.testing.assert_allclose(snapped, [[5, 0], [10, 5]])

    def test_matches_exhaustive_distance_oracle(self):
        rng = np.random.default_rng(42)
        nodes = rng.normal(size=(50, 2))
        arcs = np.column_stack([np.arange(49), np.arange(1, 50)])
        gps = rng.normal(size=(20, 2))
        ids, distances, _, _ = nearest_road(gps, nodes, arcs)
        oracle = np.array([[Point(p).distance(LineString(nodes[a]))
                            for a in arcs] for p in gps])
        np.testing.assert_allclose(distances, oracle.min(axis=1))
        np.testing.assert_array_equal(ids, oracle.argmin(axis=1))

    def test_directed_adjacency_is_not_undirected(self):
        arcs = np.array([[0, 1], [1, 2], [3, 2]])
        self.assertEqual(adjacency_rate(arcs, np.array([0, 0, 1, 2])), 2 / 3)


if __name__ == "__main__":
    unittest.main()
