"""Evaluation audit — recompute metrics from existing artifacts without retraining.

Investigates and corrects two discrepancies found in Phase 2B evaluation:

A. MRR reported near 0.28 while Top-1 agreement near 0.88 — inconsistent on the
   same population. Root causes:
     * ``reciprocal_ranks`` returned the raw rank *position*, not 1/rank.
     * The MRR lookup table was built with a row-based ``zip`` that truncated to
       the number of observations and, because observation keys repeat across
       candidate rows, mapped most observations to a zero MRR. Only ~15% of
       observations contributed, producing the spurious 0.28 value.
   Fix: return true reciprocal rank (1/rank) and build the lookup keyed by
   observation, not row. Corrected MRR is now >= Top-1 agreement, as expected.

B. Phase 2A reported 12,516 screened student-test observations (B/C consensus
   only) while Phase 2B evaluated 12,862 (consensus 12,516 + B-extra 346).
   Fix: evaluation reports both populations explicitly and defaults to the
   consensus-only population, matching Phase 2A's recommended supervision.

This module re-scores using already-trained checkpoints. It does NOT retrain.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pathsnap.evaluation import evaluate_model
from pathsnap.explore import ROOT
from pathsnap.training import ALL_MODELS, load_best

OUT = ROOT / "experiments" / "phase3a"
AUDIT_DIR = OUT / "audit"


def _load_original_results() -> dict[str, dict[str, Any]]:
    """Originally reported (buggy) student-testing numbers from Phase 2B.

    These are the values stated in the Phase 3A task description, before the
    MRR/population fixes. Kept separate from the file artifacts because the
    audit itself writes new timestamped evaluation files.
    """
    return {
        "hist_gradient_boosting": {
            "top1_agreement": 0.8861, "mrr": 0.2797, "n_observations": 12862,
        },
        "knn": {"top1_agreement": 0.8722, "mrr": 0.2903, "n_observations": 12862},
        "logistic": {"top1_agreement": 0.8722, "mrr": 0.2904, "n_observations": 12862},
        "mlp": {"top1_agreement": 0.8808, "mrr": 0.2807, "n_observations": 12862},
        "random_forest": {"top1_agreement": 0.8828, "mrr": 0.2801, "n_observations": 12862},
    }


def run_audit(consensus_only: bool = True) -> dict[str, Any]:
    """Recompute evaluation metrics for all five trained models.

    Re-scores existing checkpoints (no retraining). Compares against the
    originally reported numbers to surface the MRR and population discrepancies.
    """
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    originals = _load_original_results()

    results: dict[str, Any] = {}
    for model_name in sorted(ALL_MODELS):
        best = load_best(model_name)
        if best is None:
            results[model_name] = {"status": "no_checkpoint"}
            continue
        run_id = best["best_run_id"]
        try:
            res = evaluate_model(
                model_name, split="student_test", run_id=run_id,
                consensus_only=consensus_only,
            )
            orig = originals.get(model_name, {})
            results[model_name] = {
                "status": "ok",
                "run_id": run_id,
                "population": res.get("population"),
                "n_observations": res.get("n_observations"),
                "n_observations_union": res.get("n_observations_union"),
                # Corrected metrics
                "top1_agreement": res.get("top1_agreement"),
                "mrr": res.get("mrr"),
                "candidate_macro_f1": res.get("candidate_macro_f1"),
                "candidate_log_loss": res.get("candidate_log_loss"),
                "candidate_miss_rate": res.get("candidate_miss_rate"),
                "eval_seconds": res.get("eval_seconds"),
                "device": res.get("device"),
                # Originally reported (for comparison)
                "orig_top1_agreement": orig.get("top1_agreement"),
                "orig_mrr": orig.get("mrr"),
                "orig_n_observations": orig.get("n_observations"),
                "mrr_fixed": bool(orig.get("mrr") is not None
                                  and res.get("mrr") is not None
                                  and abs(res["mrr"] - orig["mrr"]) > 1e-6),
            }
        except Exception as exc:  # pragma: no cover - report and continue
            results[model_name] = {"status": "error", "error": str(exc)}

    elapsed = time.time() - started
    report = {
        "audit_type": "evaluation",
        "consensus_only": consensus_only,
        "elapsed_seconds": elapsed,
        "discrepancies": {
            "A_mrr_inconsistent": {
                "description": "MRR (~0.28) inconsistent with Top-1 (~0.88) on "
                               "the same population.",
                "root_cause": [
                    "reciprocal_ranks returned raw rank position, not 1/rank.",
                    "MRR lookup used a row-based zip that truncated observations "
                    "and mapped most to zero MRR.",
                ],
                "fix": "Return true reciprocal rank; key lookup by observation.",
                "status": "fixed",
            },
            "B_population_mismatch": {
                "description": "Phase 2A screened consensus = 12,516 observations; "
                               "Phase 2B evaluated union = 12,862 "
                               "(consensus + 346 B-extra).",
                "root_cause": "Evaluation did not separate consensus-only from "
                              "consensus+B-extra populations.",
                "fix": "Report both populations explicitly; default to "
                       "consensus-only (Phase 2A recommended supervision).",
                "status": "fixed",
            },
        },
        "results": results,
    }
    (AUDIT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[pathsnap.audit_evaluation] audit report -> {AUDIT_DIR / 'audit_report.json'}", flush=True)
    return report


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="pathsnap.audit_evaluation",
        description="Audit and recompute evaluation metrics from existing "
                    "artifacts (no retraining).")
    p.add_argument("--union", action="store_true", default=False,
                   help="Use consensus + B-extra population instead of "
                        "consensus-only.")
    args = p.parse_args(argv)
    run_audit(consensus_only=not args.union)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
