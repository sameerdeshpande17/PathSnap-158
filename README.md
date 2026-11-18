# PathSnap — verified Phase 1 checkpoint

Project: `/home/rohitpujari/projects/pathsnap`

Uses the existing `/home/rohitpujari/ml312/bin/python` (Python 3.12.12).
No final classifier has been trained. Exploration and a geometric nearest-road
baseline have run on all 80 training and 10 student-testing trips.
Evaluator data is absent. Nothing has been published or pushed.

## Inspect the deliverables

```bash
cd /home/rohitpujari/projects/pathsnap
xdg-open docs/phase1_report.pdf
xdg-open visualizations/train_00000083_offline.html
cat experiments/summary.json
cat experiments/integration_checks.json
cat experiments/tests.log
```

Main outputs:

- `docs/phase1_report.pdf`: reproducibly generated evidence report.
- `experiments/trip_statistics.csv`: all-trip statistics and baseline diagnostics.
- `experiments/summary.json`: partition totals and weighted summaries.
- `experiments/file_inventory.csv`: all 360 source files, sizes, rows and hashes.
- `experiments/anomaly_locations.csv`: zero-based locations of flagged events.
- `data/processed/{train,student_test}/`: observation features and baseline outputs.
- `visualizations/`: overview chart, three training-trip plots, Folium HTML maps,
  offline copies and genuine Chromium-rendered screenshots.
- `docs/data_quality.md`, `docs/methodology.md`: findings and decisions.
- `docs/requirements_checklist.csv`: official deliverables/stages and current status.
- `docs/api_contract.md`: proposed backend/frontend interface, not a running API.
- `experiments/environment.json`, `environment.freeze.txt`, `nvidia-smi.txt`:
  captured environment; full environment freeze is a record, not an install request.

## Dependencies

Existing environment was inspected first. Only missing packages were installed:
Folium, ReportLab, Pyproj, Shapely and required new Folium dependencies Branca and
xyzservices. Existing ML packages were retained. See `experiments/dependency_changes.json`.

`requirements.txt` pins direct Phase 1 packages; `requirements.lock.txt` records
their installed transitive dependency closure. Existing PyTorch 2.11.0+cu128 is
used only for GPU environment reporting, not baseline computation.
System dependencies: `pdftotext` and `pdfinfo` (Poppler), Chromium for genuine
Folium screenshots. The GPU is not required for geometry or EDA.

Check versions before installing anything:

```bash
~/ml312/bin/python -c "import importlib.metadata as m; print('\n'.join(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions()))"
# If a dependency is missing, install only that missing pinned package:
# uv pip install --python ~/ml312/bin/python PACKAGE==VERSION
```

Do not recreate or overwrite `~/ml312`. Browser assets are cached with hashes
under `visualizations/assets/`; first rendering needs internet to fetch them.
Offline HTML uses local JavaScript/CSS and supplied vector geometry; standard
HTML also offers an online OSM basemap. Keep offline HTML alongside `assets/`.

## Reproduce from original data

Original ZIP copies are preserved in `data/original/`, with SHA-256 manifest.
They came from `~/Downloads/PathSnap_student_materials.zip` and
`~/Downloads/Student folder.zip`. `prepare` verifies hashes and refuses to
overwrite modified raw files. No evaluator partition is read.

Run from this project root:

```bash
export PYTHONPATH=src
~/ml312/bin/python -m pathsnap.prepare
~/ml312/bin/python -m pathsnap.explore > experiments/exploration.log 2>&1
~/ml312/bin/python -m pathsnap.audit > experiments/audit.log 2>&1
~/ml312/bin/python -m unittest discover -s tests -v > experiments/tests.log 2>&1
~/ml312/bin/python -m pathsnap.render_maps > experiments/render_maps.log 2>&1
~/ml312/bin/python -m pathsnap.report
~/ml312/bin/python -m pathsnap.verify_outputs
pdfinfo docs/phase1_report.pdf
```

Each command must succeed before proceeding. Exploration rewrites derived
outputs only; raw inputs are preserved. Statistical outputs are deterministic;
wall-clock timing, browser rendering details and Folium element IDs can vary.
Measured initial exploration runtime is in `experiments/run.json`.

## Structure

```text
src/pathsnap/       parser, geometry baseline, EDA, audits, maps and PDF generator
data/original/     archived inputs and manifest
data/raw/          original train/student_test files
data/processed/    split-specific observation features and baseline predictions
notebooks/         notebook-oriented inspection entry point
experiments/       machine-readable evidence, hashes, versions and logs
visualizations/    genuine plots, interactive maps and browser screenshots
artifacts/         reserved for later trained assets (currently none)
tests/             parser/projection/geometric baseline verification
docs/              official PDFs, report, methodology, checklist and API contract
```

## Interpretation and incomplete work

`.route` contains a chronological arc sequence, not a GPS-aligned label vector.
The baseline uses exact projected point-to-segment distance and deterministic
lowest-ID tie-breaking. Reverse arcs can tie. Its reference-membership diagnostic
is **not accuracy**, and its displayed arc union is **not a reconstructed route**.
Do not report pointwise classifier metrics until a defensible label policy exists.

Blockers:

1. The separately supplied Phase 1 package was not located; provide its exact path
   to enable inspection and independent comparison.
2. No explicit GPS-to-reference alignment is supplied; organiser clarification
   or documented uncertainty-aware pseudo-label construction is needed.
3. Evaluator data is not supplied; it must remain unused during development.

Pending later phases: five-model training, group-CV, Bayesian optimisation,
HMM/Viterbi, model artifacts, FastAPI implementation, frontend integration and
two-laptop acceptance tests. The official first-phase rubric is not separately
specified beyond the pre-lunch evaluation notice. Team review, AI-assistance
disclosure and viva preparation remain required. GitHub publication awaits
explicit user instruction.

---

## Phase 2A — reference alignment and weak labels

All three methods have now run on **80 training + 10 student-testing trips**.
Phase 1 files are retained; verification checks their original manifest hashes
(except this intentionally extended README). No dependencies were added.

### Outputs

- `docs/phase2a_alignment_report.pdf`: measured comparison, maps and limitations.
- `docs/route_alignment_investigation.md`: official format, actual repeated-arc
  examples, all 14 adjacency breaks, conclusions and observed failures.
- `docs/alignment_methodology.md`: algorithms, constraints, screening and metrics.
- `experiments/alignment_comparison.csv`: 270 trip/method rows.
- `experiments/reference_adjacency_breaks.csv`: endpoints, geodesic separations,
  directed graph paths and nearest-GPS evidence for each break.
- `data/alignments/train/`, `data/alignments/student_test/`: per-trip A/B/C CSVs.
- `data/alignments/README.md`, `methods.json`: output schema and frozen policies.
- `experiments/alignment/summary.csv`: pooled residuals and coverage by partition.
- `experiments/alignment/training_sensitivity.csv`: training-only search-width
  and spatial-scale benchmarks.
- `experiments/alignment/training_failure_cases.csv`: diagnosed severe C failures.
- `experiments/alignment/quality_flag_counts.csv`, `gap_sensitivity.csv`,
  `method_disagreements.csv`: uncertainty and sensitivity evidence.
- `experiments/alignment/calibration.json`, `frozen_methods.json`: training-only
  provenance and frozen parameter values; no student-test fitting.
- `visualizations/alignment/`: four interactive offline Folium maps, actual browser
  screenshots, sequence plots and aggregate charts.
- `experiments/alignment/tests.log`, `verification.json`, `manifest.json`:
  automated checks and hashes.

### Inspect

```bash
cd /home/rohitpujari/projects/pathsnap
xdg-open docs/phase2a_alignment_report.pdf
xdg-open visualizations/alignment/train_00000084.html
xdg-open visualizations/alignment/student_test_00000056.html
cat experiments/alignment/summary.csv
cat experiments/alignment/verification.json
```

### Reproduce Phase 2A

Use the existing Python environment. Run stages in this order, stopping if any
command fails. These commands rewrite Phase 2A derived files, not Phase 1 outputs.
Calibration and benchmarks read training only; freeze precedes student evaluation.
Progress diagnostics are logged by trip/method and every 5,000 DP observations.

```bash
cd /home/rohitpujari/projects/pathsnap
export PYTHONPATH=src
mkdir -p experiments/alignment
~/ml312/bin/python -m unittest discover -s tests -v > experiments/alignment/tests.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_run fit > experiments/alignment_fit.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_run benchmark > experiments/alignment/benchmark.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_run freeze > experiments/alignment/freeze.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_run run > experiments/alignment/run.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_breaks > experiments/alignment_breaks.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_failures > experiments/alignment/failures.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_report > experiments/alignment/report.log 2>&1
~/ml312/bin/python -m pathsnap.alignment_verify
```

For report-only changes, `pathsnap.alignment_report --reuse-maps` uses existing
map renders. Map scripts reuse Phase 1's cached Leaflet assets, with no online
basemap or new dependency. Keep `visualizations/assets/` alongside the map folder.

### Interpretation and recommendation

- **A:** nearest reference segment, spatial lower bound but often violates order.
- **B:** monotone occurrence DP, best-supported default chronology heuristic.
- **C:** constrained probabilistic chain; severe unrecovered lock-in occurred
  on training trips `00000032` and `00000084`. Do not use its unscreened labels.
- All methods give tentative assignments to every observation; this is not
  successful validated alignment. Read quality and screened-consensus coverage.
- For exploratory weak-supervised training, use only C-export rows where
  `recommended_supervision_use` is true. B/C targets agree there and residual,
  gap, ambiguity and topology screens pass.
- No independent pointwise truth exists. Neither consensus nor downstream
  agreement with these targets establishes true accuracy/F1. Future refinement
  of C needs training-only tests of abstention/recovery; no such remedy is claimed.

Nineteen automated tests cover directed indexing, repeats, monotonicity,
dwelling, malformed routes, disconnection, timestamp gaps, projection,
exhaustive small-case DP correctness and training-only calibration guards.
Five-model training, FastAPI implementation and frontend integration remain
later tasks. Nothing has been published or pushed.

---

## Phase 2B — training infrastructure (not yet executed on full data)

A modular, resumable training pipeline is implemented under `src/pathsnap/` and
exposed through these entry points (requires `PYTHONPATH=src`):

- `python -m pathsnap.prepare_features` — cache road-candidate features
- `python -m pathsnap.train --model MODEL_NAME` — train one model
- `python -m pathsnap.train --all` — train all five models
- `python -m pathsnap.train --model MODEL_NAME --cv` — trip-grouped CV
- `python -m pathsnap.evaluate --model MODEL_NAME` — evaluate without retraining
- `python -m pathsnap.evaluate --all` — evaluate all trained models
- `python -m pathsnap.predict --model MODEL_NAME --trip TRIP_ID` — predict a trip
- `python -m pathsnap.report_models` — charts and PDF from saved outputs

No full-model training has been executed by this pipeline yet. Full training is
performed only when you explicitly invoke the `train` entry points above.

### Models (five distinct families)

1. Logistic Regression — CPU
2. K-Nearest Neighbors — CPU
3. Random Forest — CPU
4. Histogram Gradient Boosting — CPU
5. PyTorch Multi-Layer Perceptron — CUDA when available, CPU fallback

### Where to find details

- `docs/training_commands.md` — exact commands for every step
- `docs/feature_engineering.md` — 25 inference-safe features
- `docs/model_evaluation_methodology.md` — metrics and weak-label provenance
- `docs/phase2a_alignment_report.pdf`, `docs/route_alignment_investigation.md`
  — weak-label source and limitations
- `tests/test_training_infrastructure.py` — fast synthetic tests (45 total pass)

### Model artifacts

Saved under `artifacts/models/<model>/<run_id>/` with `config.json`,
`metrics.json`, `feature_schema.json`, `data_manifest.json`,
`training_log.jsonl`, `environment.json`, and the fitted pipeline (joblib) or
PyTorch `state_dict`. A `best.json` pointer names the selected run per model.

Weak labels come from frozen Phase 2A screened B/C-consensus supervision and
are **not verified ground truth**. `.route` is never read during inference.

---

## Phase 3A — evaluation audit, Bayesian optimization, final selection

Implemented (training pipeline only; full tuning runs when you invoke it):

- `python -m pathsnap.audit_evaluation` — recompute evaluation metrics from
  existing artifacts (no retraining). Fixed two defects:
  * MRR returned raw rank position (and used a row-based zip that dropped most
    observations) instead of reciprocal rank; corrected MRR is now >= Top-1.
  * Population mismatch: Phase 2A screened consensus = 12,516 obs vs Phase 2B
    union = 12,862; both are now reported, consensus-only is the default.
- `python -m pathsnap.optimize --model MODEL --trials N` — Optuna TPE
  Bayesian optimization with trip-grouped CV on training trips only.
  Models: HistGradientBoosting (primary), Random Forest, MLP (CUDA).
  Studies persist to SQLite under artifacts/optuna/ and support resume.
- `python -m pathsnap.compare_tuned` — compare completed tuning studies.
- `python -m pathsnap.select_final_model` — recommend final model from
  training-only CV evidence (no retraining, no student-test leaderboard chase).
- `python -m pathsnap.refit_final --model MODEL` — explicitly refit the
  selected model on the full training partition, save to
  artifacts/models/final/<model>/.

### Corrected student-testing results (consensus-only, 12,516 obs)

| Model | Top-1 | MRR | Macro F1 | Log loss | Device |
|-------|-------|-----|----------|----------|--------|
| HistGradientBoosting | 0.8957 | 0.9443 | 0.9278 | 0.0927 | CPU |
| Random Forest | 0.8917 | 0.9420 | 0.9278 | 0.0927 | CPU |
| MLP | 0.8889 | 0.9404 | 0.9196 | 0.1052 | CUDA |
| Logistic Regression | 0.8810 | 0.9333 | 0.9179 | 0.1195 | CPU |
| KNN | 0.8809 | 0.9337 | 0.9151 | 0.2667 | CPU |

These are weak-label agreement metrics (not verified ground-truth accuracy).

See `docs/training_commands.md` for the exact Phase 2B/3A command reference.

