"""Log-space Viterbi decoder for HMM map matching.

Decodes the most likely sequence of candidate road segments (hidden states)
given GPS observations and classifier emission probabilities.

Uses log-space arithmetic for numerical stability, supports beam pruning,
and handles unreachable transitions and large timestamp gaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pathsnap.candidates import generate, load_inputs
from pathsnap.hmm import (
    CandidateState,
    HMMSettings,
    MapMatchingHMM,
    TransitionScore,
)


@dataclass
class ViterbiResult:
    """Result of Viterbi decoding."""

    # Per-observation decoded state
    decoded_arc_ids: list[int] = field(default_factory=list)
    decoded_occurrences: list[int] = field(default_factory=list)  # candidate index within obs
    decoded_log_probs: list[float] = field(default_factory=list)
    # Full path: list of (obs_index, arc_id, candidate_index, log_prob)
    path: list[dict[str, Any]] = field(default_factory=list)
    # Diagnostics
    total_log_prob: float = 0.0
    n_observations: int = 0
    n_transitions: int = 0
    n_connected_transitions: int = 0
    n_disconnected_transitions: int = 0
    n_same_arc_dwells: int = 0
    beam_pruned_steps: int = 0
    # Per-observation candidate info for reporting
    observation_details: list[dict[str, Any]] = field(default_factory=list)


class ViterbiDecoder:
    """Log-space Viterbi decoder with optional beam pruning."""

    def __init__(self, hmm: MapMatchingHMM, settings: HMMSettings | None = None):
        self.hmm = hmm
        self.settings = settings or HMMSettings()

    # ------------------------------------------------------------------
    # Build candidate states from classifier output + candidates
    # ------------------------------------------------------------------
    def build_states(
        self,
        obs_index: int,
        gps_lon: float,
        gps_lat: float,
        gps_heading_rad: float,
        candidates_df_row_group,
        classifier_probs: np.ndarray,
    ) -> list[CandidateState]:
        """Build CandidateState objects for one observation.

        candidates_df_row_group: pandas DataFrame rows for one observation
            (must include arc_id, distance_m, projection_fraction, etc.)
        classifier_probs: array of classifier probabilities aligned with candidates_df_row_group
        """
        states = []
        max_cand = self.settings.max_candidates
        n_cand = len(candidates_df_row_group)
        limit = n_cand if max_cand is None else min(n_cand, max_cand)

        # Sort by classifier probability descending to prioritize good candidates
        order = np.argsort(-classifier_probs[:limit]) if limit > 0 else []

        for i in order:
            row = candidates_df_row_group.iloc[int(i)]
            arc_id = int(row["arc_id"])
            state = CandidateState(
                obs_index=obs_index,
                arc_id=arc_id,
                from_node=int(row.get("from_node", -1)),
                to_node=int(row.get("to_node", -1)),
                segment_length_m=float(row.get("segment_length_m", 0.0)),
                segment_heading_rad=float(row.get("road_heading_rad", 0.0)),
                projection_fraction=float(row.get("projection_fraction", 0.0)),
                distance_m=float(row["distance_m"]),
                classifier_prob=float(classifier_probs[int(i)]),
            )
            states.append(state)

        # Compute emissions
        self.hmm.compute_emissions(obs_index, states, (gps_lat, gps_lon))
        return states

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------
    def decode(
        self,
        observations: list[dict[str, Any]],
    ) -> ViterbiResult:
        """Run Viterbi decoding.

        observations: list of dicts, one per GPS observation, each containing:
            - obs_index: int
            - gps_lon, gps_lat: float
            - gps_heading_rad: float
            - dt_s: float (time since previous observation)
            - gps_displacement_m: float (distance since previous observation)
            - states: list[CandidateState] (already built with emissions)
        """
        settings = self.settings
        result = ViterbiResult()
        result.n_observations = len(observations)

        if len(observations) == 0:
            return result

        # DP tables: log_prob[obs_idx][state_idx], backpointer[obs_idx][state_idx]
        log_probs: list[list[float]] = []
        backpointers: list[list[int]] = []

        # ---- Initialization: first observation ----
        first_obs = observations[0]
        first_states = first_obs["states"]
        first_log_probs = []
        for st in first_states:
            # Uniform prior over initial states, weighted by emission
            lp = math.log(max(st.emission_prob, settings.emission_floor))
            first_log_probs.append(lp)
        log_probs.append(first_log_probs)
        backpointers.append([-1] * len(first_states))

        # Beam tracking
        prev_beam_indices = list(range(len(first_states)))
        result.beam_pruned_steps = 0

        # ---- Recursion ----
        for t in range(1, len(observations)):
            obs = observations[t]
            prev_obs = observations[t - 1]
            curr_states = obs["states"]
            prev_states = prev_obs["states"]
            dt_s = float(obs.get("dt_s", 1.0))
            gps_disp_m = float(obs.get("gps_displacement_m", 0.0))
            gps_heading = float(obs.get("gps_heading_rad", 0.0))

            curr_log_probs = []
            curr_backpointers = []

            # If no candidates for this observation, carry forward previous best
            if len(curr_states) == 0:
                log_probs.append([])
                backpointers.append([])
                result.n_transitions += 0
                continue

            # Determine which previous states to consider (beam)
            if settings.beam_width is not None and len(prev_states) > settings.beam_width:
                # Keep top-K previous states by log prob
                prev_beam_indices = sorted(
                    range(len(prev_states)),
                    key=lambda i: log_probs[t - 1][i],
                    reverse=True,
                )[: settings.beam_width]
                result.beam_pruned_steps += 1
            else:
                prev_beam_indices = list(range(len(prev_states)))

            for ci, curr_st in enumerate(curr_states):
                best_lp = -math.inf
                best_prev = -1
                best_connected = False

                for pi in prev_beam_indices:
                    prev_st = prev_states[pi]
                    trans = self.hmm.compute_transition(
                        prev_st, curr_st, dt_s, gps_disp_m, gps_heading
                    )
                    # trans.log_prob is already a log probability; add directly.
                    cand_lp = log_probs[t - 1][pi] + trans.log_prob
                    if cand_lp > best_lp:
                        best_lp = cand_lp
                        best_prev = pi
                        best_connected = trans.is_connected

                # Add emission
                best_lp += math.log(max(curr_st.emission_prob, settings.emission_floor))
                curr_log_probs.append(best_lp)
                curr_backpointers.append(best_prev)

            log_probs.append(curr_log_probs)
            backpointers.append(curr_backpointers)
            result.n_transitions += len(curr_states)

        # ---- Termination: pick best final state ----
        final_log_probs = log_probs[-1]
        best_final = int(np.argmax(final_log_probs))
        result.total_log_prob = float(final_log_probs[best_final])

        # ---- Backtrack ----
        path_states = []
        idx = best_final
        # Count connected/disconnected transitions along the chosen path
        n_connected = 0
        n_disconnected = 0
        n_dwells = 0
        for t in range(len(observations) - 1, 0, -1):
            curr_st = observations[t]["states"][idx]
            prev_idx = backpointers[t][idx]
            prev_st = observations[t - 1]["states"][prev_idx]
            dt_s = float(observations[t].get("dt_s", 1.0))
            gps_disp_m = float(observations[t].get("gps_displacement_m", 0.0))
            gps_heading = float(observations[t].get("gps_heading_rad", 0.0))
            trans = self.hmm.compute_transition(
                prev_st, curr_st, dt_s, gps_disp_m, gps_heading
            )
            if trans.is_connected:
                n_connected += 1
            else:
                n_disconnected += 1
            if prev_st.arc_id == curr_st.arc_id:
                n_dwells += 1
            idx = prev_idx

        result.n_connected_transitions = n_connected
        result.n_disconnected_transitions = n_disconnected
        result.n_same_arc_dwells = n_dwells

        path_states = []
        idx = best_final
        for t in range(len(observations) - 1, -1, -1):
            st = observations[t]["states"][idx]
            path_states.append(
                dict(
                    obs_index=t,
                    arc_id=st.arc_id,
                    candidate_index=idx,
                    log_prob=float(log_probs[t][idx]),
                    emission_prob=float(st.emission_prob),
                    distance_m=float(st.distance_m),
                    from_node=st.from_node,
                    to_node=st.to_node,
                )
            )
            idx = backpointers[t][idx] if t > 0 else -1

        path_states.reverse()
        result.path = path_states
        result.decoded_arc_ids = [p["arc_id"] for p in path_states]
        result.decoded_occurrences = [p["candidate_index"] for p in path_states]
        result.decoded_log_probs = [p["log_prob"] for p in path_states]

        # Observation details
        for t, obs in enumerate(observations):
            result.observation_details.append(
                dict(
                    obs_index=t,
                    n_candidates=len(obs["states"]),
                    decoded_arc_id=result.decoded_arc_ids[t],
                    decoded_log_prob=result.decoded_log_probs[t],
                    candidate_arc_ids=[s.arc_id for s in obs["states"]],
                    candidate_probs=[s.emission_prob for s in obs["states"]],
                )
            )

        return result
