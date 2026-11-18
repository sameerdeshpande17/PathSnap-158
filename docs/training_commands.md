# PathSnap Phase 2B — Training Commands Manual

All commands run with the existing environment:

```
/home/rohitpujari/ml312/bin/python
```

and require `PYTHONPATH=src` (or install the package) so `pathsnap` is importable:

```bash
cd /home/rohitpujari/projects/pathsnap
export PYTHONPATH=src
```

No CUDA toolkit is installed and no virtual environment is recreated. PyTorch 2.11.0+cu128 already works with CUDA.

## 1. Prepare and cache candidate features

Generate STRtree-based road candidates and 25 inference-safe features for every GPS observation. Cache keys reflect dataset identity, feature configuration, and alignment version.

```bash
# Cache both train and student-testing splits (default)
/home/rohitpujari/ml312/bin/python -m pathsnap.prepare_features

# One split only
/home/rohitpujari/ml312/bin/python -m pathsnap.prepare_features --split train
/home/rohitpujari/ml312/bin/python -m pathsnap.prepare_features --split student_test

# Custom radius / candidate cap, force rebuild
/home/rohitpujari/ml312/bin/python -m pathsnap.prepare_features \
    --radius-m 75 --max-candidates 24 --fallback nearest --force
```

Outputs are written to `data/candidates/cache/<key>.parquet` and inspected via the cache manifest. Candidate generation never reads `.route`.

## 2. Train one model

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model logistic
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model knn
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model random_forest
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model hist_gradient_boosting
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model mlp
```

Options:

- `--strategy consensus` (default) or `consensus_plus_B`
- `--cv`: run trip-grouped cross-validation (3 folds) instead of final training
- `--skip-existing`: skip if a compatible checkpoint already exists
- `--force`: create a new run even if compatible checkpoint exists (never overwrites)
- `--resume RUN_ID`: resume an interrupted PyTorch MLP run

Example (cross-validation):

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model random_forest --cv
```

## 3. Train all five models sequentially

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.train --all
```

Models are trained one at a time. The MLP uses CUDA when available; scikit-learn models run on CPU. Each run writes to a unique directory under `artifacts/models/<model>/<run_id>/`.

## 4. Monitor CPU and GPU

- The active device is printed at the start of every run (`CPU` or `cuda`).
- scikit-learn estimators run on CPU (they have no CUDA device parameter).
- The MLP uses CUDA when `torch.cuda.is_available()`; otherwise CPU fallback (device is never silently changed).
- GPU memory-safe batching is configured via `batch_size` and `inference_batch_size` in `experiments/phase2b/config.json`.
- Monitor externally with `nvidia-smi` (GPU) and `htop`/`ps` (CPU/RSS).

## 5. Resume supported training runs

Resume applies to the PyTorch MLP (optimizer, scheduler, epoch, RNG, early-stopping states are saved). scikit-learn models do not support reliable continuation.

```bash
# Skip if compatible checkpoint exists
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model mlp --skip-existing

# Force a fresh run
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model mlp --force

# Resume a specific interrupted run
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model mlp --resume mlp_consensus_20261118-120000_abcd1234
```

List existing runs:

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.train --model mlp --help  # inspect options
/home/rohitpujari/ml312/bin/python -c "from pathsnap.training import list_runs; [print(r) for r in list_runs('mlp')]"
```

## 6. Load saved checkpoints

Every run directory contains:

```
artifacts/models/<model>/<run_id>/
  pipeline.joblib          (scikit-learn)  OR  model.pt + metadata.json  (PyTorch)
  config.json
  metrics.json
  feature_schema.json
  data_manifest.json
  training_log.jsonl
  environment.json
```

A `best.json` pointer in `artifacts/models/<model>/` names the selected run.

Evaluation and prediction never trigger training automatically.

## 7. Evaluate trained models

```bash
# Best checkpoint on student-testing (default)
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate --model random_forest

# Specific run
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate --model mlp --run-id <RUN_ID>

# On training split
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate --model logistic --split train

# All available models
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate --all
```

Results are saved to `artifacts/models/evaluation/`.

## 8. Run predictions on a trip

Prediction reads only track/nodes/arcs — never `.route`:

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.predict \
    --model random_forest --trip 00000056

/home/rohitpujari/ml312/bin/python -m pathsnap.predict \
    --model mlp --trip 00000056 --run-id <RUN_ID>
```

Top-1 results are printed and exported to `artifacts/models/predictions/`.

## 9. Generate comparison report

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.report_models
/home/rohitpujari/ml312/bin/python -m pathsnap.report_models --no-pdf
/home/rohitpujari/ml312/bin/python -m pathsnap.report_models --output docs/phase2b_model_report.pdf
```

Charts go to `visualizations/phase2b/`; PDF to `docs/phase2b_model_report.pdf`.

## Expected CPU/GPU usage per model

| Model | Device | Notes |
|---|---|---|
| Logistic Regression | CPU | `lbfgs`, fast, small memory |
| K-Nearest Neighbors | CPU | `ball_tree`; prediction cost grows with training set size |
| Random Forest | CPU | 100 trees, `n_jobs=4`, moderate memory |
| HistGradientBoosting | CPU | 150 iterations, no early stopping by default |
| MLP | CUDA (or CPU fallback) | 2-layer, epoch logging, early stopping, best-checkpoint restore |

## Weak-supervision provenance

Targets come from the frozen Phase 2A screened B/C-consensus labels. They are **weak/inferred supervision**, not verified ground truth. No model training has been executed by this pipeline yet — run the commands above when ready.

---

# Phase 3B — HMM/Viterbi route reconstruction, FastAPI backend

## 10. HMM/Viterbi route reconstruction

Decode GPS trips into connected road-arc sequences using the trained classifier + HMM + Viterbi:

```bash
# Decode a single trip (no .route accessed)
/home/rohitpujari/ml312/bin/python -m pathsnap.decode_route \
    --model hist_gradient_boosting --trip 00000056 --split student_test

# Save output to a file
/home/rohitpujari/ml312/bin/python -m pathsnap.decode_route \
    --model hist_gradient_boosting --trip 00000056 --split student_test \
    --output /tmp/trip_00000056.json
```

## 11. Route-level evaluation

Evaluate classifier vs HMM/Viterbi predictions against reference routes:

```bash
# Evaluate a single trip
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate_routes \
    --model hist_gradient_boosting --trip 00000056 --split student_test \
    --output-csv experiments/phase3b/route_eval_00000056.csv \
    --output-json experiments/phase3b/route_eval_00000056.json

# Evaluate all 10 student-testing trips
/home/rohitpujari/ml312/bin/python -m pathsnap.evaluate_routes \
    --model hist_gradient_boosting --split student_test \
    --output-csv experiments/phase3b/route_evaluation.csv \
    --output-json experiments/phase3b/route_evaluation.json
```

## 12. Route report

Generate charts and PDF from saved route evaluation outputs:

```bash
/home/rohitpujari/ml312/bin/python -m pathsnap.report_routes
/home/rohitpujari/ml312/bin/python -m pathsnap.report_routes --no-pdf
/home/rohitpujari/ml312/bin/python -m pathsnap.report_routes \
    --output docs/phase3b_route_report.pdf
```

Charts go to `visualizations/phase3b/`; PDF to `docs/phase3b_route_report.pdf`.

## 13. FastAPI backend

Start the server:

```bash
cd /home/rohitpujari/projects/pathsnap
export PYTHONPATH=src
/home/rohitpujari/ml312/bin/python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, model readiness, CUDA availability |
| `/api/model` | GET | Selected model metadata and inference configuration |
| `/api/trips` | GET | List available trips for a split |
| `/api/predict` | POST | Full map-matching pipeline |
| `/api/evaluate` | POST | Route-level evaluation vs reference route |

Example:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "00000056", "split": "student_test"}'

curl http://localhost:8000/health
curl http://localhost:8000/api/model
curl "http://localhost:8000/api/trips?split=student_test"
```

OpenAPI docs: `http://localhost:8000/docs`

## Expected CPU/GPU usage per model

| Model | Device | Notes |
|---|---|---|
| Logistic Regression | CPU | `lbfgs`, fast, small memory |
| K-Nearest Neighbors | CPU | `ball_tree`; prediction cost grows with training set size |
| Random Forest | CPU | 100 trees, `n_jobs=4`, moderate memory |
| HistGradientBoosting | CPU | 150 iterations, no early stopping by default |
| MLP | CUDA (or CPU fallback) | 2-layer, epoch logging, early stopping, best-checkpoint restore |

## Weak-supervision provenance

Targets come from the frozen Phase 2A screened B/C-consensus labels. They are **weak/inferred supervision**, not verified ground truth. No model training has been executed by this pipeline yet — run the commands above when ready.
