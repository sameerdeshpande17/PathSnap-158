# PathSnap Phase 3A — Evaluation Audit, Bayesian Optimization, Final Selection

All commands run with the existing environment:

```bash
cd /home/rohitpujari/projects/pathsnap
export PYTHONPATH=src
/home/rohitpujari/ml312/bin/python -m pathsnap <command> [options]
```

No full tuning, refitting, or expensive experiments run automatically. They
execute only when you invoke the commands below.

## 1. Evaluation audit (run first — no retraining)

Recompute evaluation metrics from existing trained checkpoints. This does not
retrain any model.

```bash
# Consensus-only population (Phase 2A recommended supervision; default)
/home/rohitpujari/ml312/bin/python -m pathsnap.audit_evaluation

# Union population (consensus + B-extra, larger)
/home/rohitpujari/ml312/bin/python -m pathsnap.audit_evaluation --union
```

What it does:
- Re-scores all five trained models on student-testing (12,516 consensus obs).
- Reports Top-1 agreement, MRR, Macro F1, log loss, miss rate, device, runtime.
- Compares against the originally reported (buggy) numbers.
- Writes `experiments/phase3a/audit/audit_report.json`.

### Defects found and fixed (in src/pathsnap/phase2b_metrics.py)

| # | Discrepancy | Root cause | Fix |
|---|-------------|------------|-----|
| A | MRR ~0.28 vs Top-1 ~0.88 (inconsistent) | `reciprocal_ranks` returned raw rank position, not 1/rank; MRR lookup used row-based zip truncation that dropped ~85% of observations | Return true reciprocal rank (1/rank); key lookup by observation, not row |
| B | 12,516 (Phase 2A consensus) vs 12,862 (Phase 2B union) | Evaluation mixed consensus-only and consensus+B-extra populations | Report both explicitly; default to consensus-only |

### Corrected student-testing results (consensus-only, 12,516 obs)

| Model | Top-1 | MRR | Macro F1 | Log loss | Device |
|-------|-------|-----|----------|----------|--------|
| HistGradientBoosting | 0.8957 | 0.9443 | 0.9278 | 0.0927 | CPU |
| Random Forest | 0.8917 | 0.9420 | 0.9278 | 0.0927 | CPU |
| MLP | 0.8889 | 0.9404 | 0.9196 | 0.1052 | CUDA |
| Logistic Regression | 0.8810 | 0.9333 | 0.9179 | 0.1195 | CPU |
| KNN | 0.8809 | 0.9337 | 0.9151 | 0.2667 | CPU |

Weak-label agreement, not verified ground-truth accuracy.

## 2. Bayesian optimization (Optuna TPE)

Run only when you want to tune. Trip-grouped CV strictly within the 80 training
trips. Student-testing never used for tuning, trials, calibration, or selection.

```bash
# HistGradientBoosting (primary model), 50 trials
/home/rohitpujari/ml312/bin/python -m pathsnap.optimize \
    --model hist_gradient_boosting --trials 50

# Random Forest, 30 trials
/home/rohitpujari/ml312/bin/python -m pathsnap.optimize \
    --model random_forest --trials 30

# MLP (CUDA), 30 trials
/home/rohitpujari/ml312/bin/python -m pathsnap.optimize \
    --model mlp --trials 30
```

Options:
- `--strategy consensus` (default) or `consensus_plus_B`
- `--seed SEED` (default 42)
- `--folds N` (default 3, trip-grouped)
- `--no-resume` (start a fresh SQLite study)
- `--config PATH` (experiment config JSON)

Studies persist to SQLite under `artifacts/optuna/` (`phase3a_<model>_<strategy>.db`).
Each trial saves: model, trial number, hyperparameters, per-fold CV results,
mean/std, duration, seed, data/feature version, pruned/failed status. Studies
resume after interruption (`load_if_exists=True`). No checkpoint is overwritten.

Search spaces:
- HistGradientBoosting: learning_rate, max_iter, max_leaf_nodes, max_depth,
  min_samples_leaf, l2_regularization, max_bins
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf,
  max_features, class_weight
- MLP: hidden layer dims, n_layers, dropout, learning_rate, weight_decay,
  batch_size, epochs (CUDA when available, CPU fallback)

Objective: mean CV Top-1 weak-label agreement (trip-macro). Candidate log loss
and per-trip stability are secondary diagnostics. Candidate-level binary
accuracy is deliberately NOT the objective.

## 3. Compare tuned models

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.compare_tuned
```

Reads all completed Optuna study summaries, prints and saves
`experiments/phase3a/comparison_tuned.csv` with best Top-1, std, log loss,
stability, pruned/failed trials, wall time, and hyperparameters per model.

## 4. Select final model

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.select_final_model
```

Recommends the best model from training-only CV evidence (mean Top-1 primary;
std/stability and log loss as tie-breakers; <=1% relative improvement prefers
the simpler model). Does NOT retrain. Writes
`experiments/phase3a/selected_model.json`.

## 5. Refit final model (explicit, separate command)

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.refit_final --model hist_gradient_boosting
```

Refits the selected model with its best hyperparameters on the full training
partition (all 80 trips), saves the final checkpoint to
`artifacts/models/final/<model>/` with:
- fitted model (`pipeline.joblib` or `model.pt`)
- `config.json`, `metrics.json`, `feature_schema.json`,
  `data_manifest.json`, `training_log.jsonl`, `environment.json`, `checksum.json`

Reload + deterministic prediction equivalence verified. Inference never reads
`.route`.

## 6. Weak-supervision limitations (documented)

All metrics measure agreement with Phase 2A weak B/C-consensus labels, which
themselves are inferred from `.route` alignment (not independently verified
ground truth). True pointwise accuracy cannot be computed from the supplied
data because `.route` has no explicit GPS-to-arc alignment. Route-level
diagnostics and a manual-review workflow remain for future work.
