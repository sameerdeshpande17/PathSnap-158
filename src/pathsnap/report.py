"""Build a PDF checkpoint report from saved evidence, without invented metrics."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, LongTable,
)

from .explore import ROOT


def main():
    summary = json.loads((ROOT / "experiments/summary.json").read_text())
    env = json.loads((ROOT / "experiments/environment.json").read_text())
    run = json.loads((ROOT / "experiments/run.json").read_text())
    checks = json.loads((ROOT / "experiments/integration_checks.json").read_text())
    maps = json.loads((ROOT / "experiments/representative_maps.json").read_text())
    stats = pd.read_csv(ROOT / "experiments/trip_statistics.csv", dtype={"trip_id": str})
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    story, markdown = [], []

    def text(value, style="BodyText"):
        story.append(Paragraph(escape(str(value)), styles[style]))
        story.append(Spacer(1, 6))
        markdown.append(("# " if style == "Title" else "## " if style == "Heading1" else
                         "### " if style == "Heading2" else "") + str(value) + "\n")

    def image(path, caption):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            width, height = im.size
        w = 6.7 * inch
        h = w * height / width
        story.append(Image(str(path), width=w, height=h))
        text(caption)
        markdown.append(f"![{caption}](../{path.relative_to(ROOT)})\n")

    def table(rows, widths=None):
        formatted = [[Paragraph(escape(str(c)), styles["BodyText"]) for c in row]
                     for row in rows]
        t = LongTable(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f2")),
            ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        markdown.extend([" | ".join(map(str, row)) + "\n" for row in rows])

    def document(path):
        buffer = []
        def flush():
            if buffer:
                text(" ".join(buffer))
                buffer.clear()
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                flush()
                text(line.lstrip("# "), "Heading2")
            elif not line.strip():
                flush()
            else:
                buffer.append(line.strip())
        flush()

    text("PathSnap - Phase 1 Evidence Report", "Title")
    text("Dataset exploration, reference-format investigation and geometric baseline")
    text("Checkpoint generated from local experiment artifacts. No trained classifier, "
         "Bayesian optimisation, HMM, FastAPI deployment or UI demonstration is claimed.")
    text("AI assistance: code and documentation were developed with an AI coding agent. "
         "Experiments were executed locally; team review and understanding remain required.")
    text("1. Scope, inputs and reproducibility", "Heading1")
    text("All five original official guideline PDFs were read, including assessment and "
         "plagiarism policy. See docs/official for originals and extracted text. The documents "
         "mention a pre-lunch first evaluation without a separate detailed Phase 1 rubric.")
    for item in json.loads((ROOT / "data/original/manifest.json").read_text()):
        text(f"Input: {item['archive']}; SHA-256: {item['sha256']}")
    text(f"Python: {env['python'].splitlines()[0]}; executable: {env['executable']}.")
    text(f"PyTorch: {env['torch']}; CUDA available: {env['cuda_available']}; GPU: {env['gpu']}.")
    text(f"Exploration wall time: {run['elapsed_seconds']:.3f}s for "
         f"{run['completed_trips']} trips; failed trips: {run['failed_trips']}; models trained: 0.")
    text("Reproduce from the project root using README.md. Commands and raw logs are "
         "retained under experiments/. Node and arc counts are sums of trip-local "
         "records, not globally unique roads.")
    text("2. Dataset understanding and exploratory data analysis", "Heading1")
    fields = [
        ("Trips", "trips"), ("GPS observations", "gps_count"), ("Node records", "node_count"),
        ("Directed arc records", "arc_count"), ("Reference arc occurrences", "route_count"),
        ("Missing numeric values", "missing_values"), ("Nonpositive time deltas", "nonpositive_dt"),
        ("Gaps >5 seconds", "gaps_over_5s"), ("Maximum gap (seconds)", "max_gap_s"),
        ("Reference adjacency breaks", "route_breaks"),
        ("Zero-length arcs", "zero_length_arcs"),
    ]
    table([["Measured quantity", "Training", "Student testing"]] +
          [[title, f"{summary['train'][key]:,}", f"{summary['student_test'][key]:,}"]
           for title, key in fields], [240, 120, 120])
    text("Exactly-one-second step fractions: training "
         f"{summary['train']['sampling_1s_fraction']:.4%}; student testing "
         f"{summary['student_test']['sampling_1s_fraction']:.4%}. "
         "The evaluator partition was not supplied and was not accessed.")
    image(ROOT / "visualizations/dataset_overview.png",
          "Figure 1. All available trips; partitions shown separately. Speed is inferred from GPS.")
    document(ROOT / "docs/data_quality.md")
    text("2.1 Methodology, feature rationale and label limitations", "Heading1")
    document(ROOT / "docs/methodology.md")
    text("3. Geometric baseline (not model training)", "Heading1")
    table([["Diagnostic", "Training", "Student testing"]] + [
        [label, f"{summary['train'][key]:.4f}", f"{summary['student_test'][key]:.4f}"]
        for label, key in [
            ("Nearest distance median (m)", "nearest_distance_median_m"),
            ("Nearest distance 95th percentile (m)", "nearest_distance_p95_m"),
            ("Direct adjacency, micro fraction", "baseline_direct_adjacency_micro"),
            ("Reference membership, NOT accuracy", "baseline_reference_membership_micro"),
            ("Baseline computation total (seconds)", "baseline_total_seconds"),
        ]], [240, 120, 120])
    text("No observation-level Top-1, Macro F1, precision, recall or log loss is claimed: "
         "the supplied route sequence has no explicit timestamp alignment. Reference "
         "membership ignores route order; direct adjacency is not full route reachability.")
    text("3.1 Five-model experimentation and comparison - pending", "Heading2")
    text("No final model was trained, per Phase 1 scope. Later work must record every "
         "model's rationale, configuration, objective, fold and student-test results, "
         "overfitting evidence, timings and genuine comparison plots.")
    text("4. Bayesian hyperparameter optimisation - pending", "Heading1")
    text("No trials performed. Use training-only group-aware cross-validation; retain "
         "search space rationale, objective, trials, best parameters, cost and tuned/untuned comparison.")
    text("5. HMM and Viterbi route reconstruction - pending", "Heading1")
    text("No HMM fitted or decoded. Candidate-state mapping, emission calibration, "
         "directed transitions, gap handling and classifier-only comparisons are planned.")
    text("6. Geographical visualisations", "Heading1")
    for entry in maps:
        story.append(PageBreak())
        text(f"Training trip {entry['trip_id']}: {entry['reason']}", "Heading2")
        stem = ROOT / "visualizations" / f"train_{entry['trip_id']}"
        image(stem.with_name(stem.name + "_folium").with_suffix(".png"),
              "Actual Chromium render of the generated Folium vector map. Blue GPS, green "
              "reference arcs, orange nearest-road baseline, grey network. Offline view has "
              "no raster basemap; full HTML provides optional OpenStreetMap tiles.")
        text(f"Displayed road arcs: {entry['displayed_road_arcs']:,} of "
             f"{entry['total_road_arcs']:,}; clipped to GPS/reference extent. "
             f"Interactive HTML: visualizations/train_{entry['trip_id']}.html")
        image(stem.with_suffix(".png"), "Companion static longitude/latitude plot of the same supplied geometry.")
    text("7. API, UI and deployment - contract only", "Heading1")
    text("docs/api_contract.md defines the two-laptop interface, schemas, failure cases, "
         "server-side evaluation isolation and integration acceptance criteria. No working "
         "API/UI screenshots are presented because these components are not implemented.")
    text("8. Verification and incomplete tasks", "Heading1")
    text((ROOT / "experiments/tests.log").read_text())
    text(f"Integration checks: {checks['archive_members_hash_verified']} archive member hashes; "
         f"{checks['trip_output_row_counts_verified']} output row-count checks; "
         f"{checks['baseline_arc_bounds_verified']} baseline arc-bound checks.")
    text("Maximum relative projected/geodesic arc-length discrepancy across checked nontrivial "
         f"arcs: {checks['max_projection_relative_arc_length_error']:.8%}. "
         "This measures endpoint-distance distortion, not nearest-road geodesic equivalence.")
    text("Blockers: supplied Phase 1 package not located; observation-to-reference alignment "
         "not supplied; evaluator dataset absent. Pending by design: five models, group-CV, "
         "Bayesian search, HMM, trained artifacts, API/UI and cross-laptop demonstration. "
         "Team review/viva preparation and eventual authorised GitHub publication remain outstanding.")
    text("8.1 Requirements coverage", "Heading2")
    checklist = list(csv.DictReader((ROOT / "docs/requirements_checklist.csv").open()))
    table([["Official requirement", "Stage/status", "Evidence or next step"]] +
          [[f"{r['source']} {r['section']}: {r['requirement']}",
            f"{r['stage']}: {r['status']}", r["evidence_or_next_step"]] for r in checklist],
          [210, 75, 195])
    story.append(PageBreak())
    text("Appendix: all-trip measurements", "Heading1")
    table([["Split / trip", "GPS", "Nodes", "Arcs", "Max gap s", "Route breaks"]] +
          [[f"{r.split} / {r.trip_id}", r.gps_count, r.node_count, r.arc_count,
            f"{r.max_gap_s:g}", r.route_breaks] for r in stats.itertuples()],
          [135, 55, 80, 80, 70, 60])
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 22, f"PathSnap Phase 1 | locally measured evidence | page {doc.page}")
    SimpleDocTemplate(str(ROOT / "docs/phase1_report.pdf"), pagesize=(612, 792),
                      rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
                      title="PathSnap Phase 1 Evidence Report",
                      author="PathSnap team - AI-assisted checkpoint").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "docs/phase1_report.md").write_text("\n".join(markdown))
    print("Generated docs/phase1_report.pdf and docs/phase1_report.md")


if __name__ == "__main__":
    main()
