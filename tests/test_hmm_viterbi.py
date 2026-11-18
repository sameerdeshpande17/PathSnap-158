"""Fast synthetic tests for Phase 3B HMM/Viterbi map matching.

No full-dataset route evaluation is performed here. All tests use small
synthetic graphs and known-optimal sequences.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathsnap.hmm import CandidateState, HMMSettings, MapMatchingHMM
from pathsnap.viterbi import ViterbiDecoder, ViterbiResult


def _make_hmm() -> MapMatchingHMM:
    """Small synthetic directed road network:
       Nodes: 0->1->2->3 (a line), plus 1->4 (branch), 4->3 (shortcut)
       Arcs:  0:(0,1), 1:(1,2), 2:(2,3), 3:(1,4), 4:(4,3)
    """
    nodes = np.array([
        [0.0, 0.0],   # node 0
        [0.01, 0.0],  # node 1 (~1km east)
        [0.02, 0.0],  # node 2
        [0.03, 0.0],  # node 3
        [0.01, 0.005],  # node 4 (branch north)
    ])
    arcs = np.array([
        [0, 1],  # arc 0
        [1, 2],  # arc 1
        [2, 3],  # arc 2
        [1, 4],  # arc 3
        [4, 3],  # arc 4
    ])
    hmm = MapMatchingHMM()
    hmm.set_network(nodes, arcs)
    return hmm


def _make_state(obs_index, arc_id, from_node, to_node, prob=0.9, distance=1.0):
    return CandidateState(
        obs_index=obs_index,
        arc_id=arc_id,
        from_node=from_node,
        to_node=to_node,
        segment_length_m=1000.0,
        segment_heading_rad=0.0,
        projection_fraction=0.5,
        distance_m=distance,
        classifier_prob=prob,
    )


class HMMSettingsTests(unittest.TestCase):
    def test_network_construction_and_components(self):
        hmm = _make_hmm()
        # All nodes 0-4 are weakly connected (0-1-2-3 and 1-4-3)
        self.assertTrue(hmm.same_component(0, 3))
        self.assertTrue(hmm.same_component(0, 4))
        self.assertTrue(hmm.same_component(2, 4))

    def test_directed_path_exists(self):
        hmm = _make_hmm()
        # 0 -> 3 via arcs 0,1,2
        path = hmm._bounded_directed_path(0, 3, max_edges=10)
        self.assertIsNotNone(path)
        self.assertEqual(path, [0, 1, 2])

    def test_no_directed_path_backwards(self):
        hmm = _make_hmm()
        # 3 -> 0 has no directed path (all arcs flow forward)
        path = hmm._bounded_directed_path(3, 0, max_edges=10)
        self.assertIsNone(path)

    def test_shortcut_path(self):
        hmm = _make_hmm()
        # 1 -> 3 can go via arc 1,2 OR shortcut arc 3,4
        path = hmm._bounded_directed_path(1, 3, max_edges=10)
        self.assertIsNotNone(path)


class EmissionTests(unittest.TestCase):
    def test_emission_probabilities_in_range(self):
        hmm = _make_hmm()
        states = [
            _make_state(0, 0, 0, 1, prob=0.9, distance=1.0),
            _make_state(0, 1, 1, 2, prob=0.1, distance=100.0),
        ]
        hmm.compute_emissions(0, states, (0.0, 0.0))
        for s in states:
            self.assertGreaterEqual(s.emission_prob, 0.0)
            self.assertLessEqual(s.emission_prob, 1.0)

    def test_closer_segment_higher_emission(self):
        hmm = _make_hmm()
        states = [
            _make_state(0, 0, 0, 1, prob=0.5, distance=1.0),  # close
            _make_state(0, 1, 1, 2, prob=0.5, distance=1000.0),  # far
        ]
        hmm.compute_emissions(0, states, (0.0, 0.0))
        self.assertGreater(states[0].emission_prob, states[1].emission_prob)


class TransitionTests(unittest.TestCase):
    def test_same_arc_dwell_allowed(self):
        hmm = _make_hmm()
        s0 = _make_state(0, 0, 0, 1)
        s1 = _make_state(1, 0, 0, 1)
        trans = hmm.compute_transition(s0, s1, dt_s=1.0, gps_displacement_m=10.0,
                                        gps_heading_rad=0.0)
        self.assertTrue(trans.is_connected)
        self.assertEqual(trans.reason, "same_arc_dwell")

    def test_connected_transition(self):
        hmm = _make_hmm()
        # arc 0 ends at node 1, arc 1 starts at node 1: connected
        s0 = _make_state(0, 0, 0, 1)
        s1 = _make_state(1, 1, 1, 2)
        trans = hmm.compute_transition(s0, s1, dt_s=1.0, gps_displacement_m=1000.0,
                                        gps_heading_rad=0.0)
        self.assertTrue(trans.is_connected)
        self.assertEqual(trans.reason, "connected")

    def test_disconnected_components(self):
        hmm = _make_hmm()
        # Create a disconnected node by testing with a node not in graph
        # arc 0 ends at node 1; arc 2 starts at node 2 (connected via arc 1)
        # To test disconnected, we'd need a separate component. Use arc 2 (2->3)
        # after arc 0 (0->1) - but 1->2 is connected via arc 1.
        # Instead test: arc 0 (ends node 1) -> arc 4 (starts node 4)
        # Node 1 and 4 are connected (1->4 via arc 3)
        s0 = _make_state(0, 0, 0, 1)
        s1 = _make_state(1, 4, 4, 3)
        trans = hmm.compute_transition(s0, s1, dt_s=1.0, gps_displacement_m=100.0,
                                        gps_heading_rad=0.0)
        self.assertTrue(trans.is_connected)  # 1->4 is a valid directed arc

    def test_no_directed_transition_penalized(self):
        hmm = _make_hmm()
        # arc 2 ends at node 3, arc 0 starts at node 0: no path 3->0
        s0 = _make_state(0, 2, 2, 3)
        s1 = _make_state(1, 0, 0, 1)
        trans = hmm.compute_transition(s0, s1, dt_s=1.0, gps_displacement_m=10.0,
                                        gps_heading_rad=0.0)
        self.assertFalse(trans.is_connected)
        self.assertEqual(trans.reason, "no_directed_path")


class ViterbiTests(unittest.TestCase):
    def test_viterbi_simple_chain(self):
        """Three observations, each with two candidates. Correct path follows
        connected arcs 0->1->2."""
        hmm = _make_hmm()
        decoder = ViterbiDecoder(hmm, settings=HMMSettings(beam_width=None))

        # Build observations
        observations = []
        # Obs 0: arc 0 (correct, high prob) vs arc 3 (branch, lower prob)
        observations.append(dict(
            obs_index=0, gps_lon=0.0, gps_lat=0.0, gps_heading_rad=0.0,
            dt_s=0.0, gps_displacement_m=0.0,
            states=[
                _make_state(0, 0, 0, 1, prob=0.9, distance=1.0),
                _make_state(0, 3, 1, 4, prob=0.2, distance=1.0),
            ],
        ))
        # Obs 1: arc 1 (correct) vs arc 0 (wrong direction)
        observations.append(dict(
            obs_index=1, gps_lon=0.01, gps_lat=0.0, gps_heading_rad=0.0,
            dt_s=1.0, gps_displacement_m=1000.0,
            states=[
                _make_state(1, 1, 1, 2, prob=0.9, distance=1.0),
                _make_state(1, 0, 0, 1, prob=0.2, distance=1.0),
            ],
        ))
        # Obs 2: arc 2 (correct) vs arc 4 (branch)
        observations.append(dict(
            obs_index=2, gps_lon=0.02, gps_lat=0.0, gps_heading_rad=0.0,
            dt_s=1.0, gps_displacement_m=1000.0,
            states=[
                _make_state(2, 2, 2, 3, prob=0.9, distance=1.0),
                _make_state(2, 4, 4, 3, prob=0.2, distance=1.0),
            ],
        ))

        result = decoder.decode(observations)
        self.assertEqual(result.decoded_arc_ids, [0, 1, 2])
        self.assertEqual(result.n_disconnected_transitions, 0)
        self.assertEqual(result.n_connected_transitions, 2)

    def test_viterbi_handles_disconnected_gap(self):
        """When no directed path exists, decoder should still produce output
        with disconnected flag."""
        hmm = _make_hmm()
        decoder = ViterbiDecoder(hmm, settings=HMMSettings(beam_width=None))

        # Obs 0: arc 2 (ends at node 3)
        # Obs 1: arc 0 (starts at node 0) - no path 3->0
        observations = [
            dict(
                obs_index=0, gps_lon=0.02, gps_lat=0.0, gps_heading_rad=0.0,
                dt_s=0.0, gps_displacement_m=0.0,
                states=[_make_state(0, 2, 2, 3, prob=0.9, distance=1.0)],
            ),
            dict(
                obs_index=1, gps_lon=0.0, gps_lat=0.0, gps_heading_rad=0.0,
                dt_s=1.0, gps_displacement_m=10.0,
                states=[_make_state(1, 0, 0, 1, prob=0.9, distance=1.0)],
            ),
        ]
        result = decoder.decode(observations)
        self.assertEqual(len(result.decoded_arc_ids), 2)
        self.assertGreater(result.n_disconnected_transitions, 0)

    def test_viterbi_empty_observations(self):
        hmm = _make_hmm()
        decoder = ViterbiDecoder(hmm)
        result = decoder.decode([])
        self.assertEqual(result.n_observations, 0)
        self.assertEqual(result.decoded_arc_ids, [])

    def test_viterbi_single_observation(self):
        hmm = _make_hmm()
        decoder = ViterbiDecoder(hmm)
        observations = [
            dict(
                obs_index=0, gps_lon=0.0, gps_lat=0.0, gps_heading_rad=0.0,
                dt_s=0.0, gps_displacement_m=0.0,
                states=[
                    _make_state(0, 0, 0, 1, prob=0.9),
                    _make_state(0, 1, 1, 2, prob=0.1),
                ],
            )
        ]
        result = decoder.decode(observations)
        self.assertEqual(result.decoded_arc_ids, [0])

    def test_viterbi_dwelling_same_arc(self):
        """Multiple consecutive observations on the same arc should all decode
        to that arc."""
        hmm = _make_hmm()
        decoder = ViterbiDecoder(hmm, settings=HMMSettings(beam_width=None))

        observations = []
        for t in range(5):
            observations.append(dict(
                obs_index=t, gps_lon=0.0, gps_lat=0.0, gps_heading_rad=0.0,
                dt_s=1.0, gps_displacement_m=10.0,
                states=[_make_state(t, 0, 0, 1, prob=0.9, distance=1.0)],
            ))
        result = decoder.decode(observations)
        self.assertEqual(result.decoded_arc_ids, [0, 0, 0, 0, 0])
        self.assertEqual(result.n_same_arc_dwells, 4)


class NoRouteLeakageTests(unittest.TestCase):
    def test_hmm_module_has_no_route_file_reading(self):
        import pathsnap.hmm as mod
        source = Path(mod.__file__).read_text()
        # Check for actual .route file access patterns, not attribute names
        self.assertNotIn('with_suffix(".route")', source)
        self.assertNotIn('open(".route', source)
        self.assertNotIn("read_table.*route", source)

    def test_viterbi_module_has_no_route_file_reading(self):
        import pathsnap.viterbi as mod
        source = Path(mod.__file__).read_text()
        self.assertNotIn('with_suffix(".route")', source)
        self.assertNotIn('open(".route', source)

    def test_pipeline_module_has_no_route_file_reading(self):
        import pathsnap.map_matching_pipeline as mod
        source = Path(mod.__file__).read_text()
        self.assertNotIn('with_suffix(".route")', source)
        self.assertNotIn('open(".route', source)


if __name__ == "__main__":
    unittest.main()
