#!/usr/bin/env python3
"""Generate Phase 3B route evaluation report PDF."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

ROOT = Path(__file__).resolve().parents[2]
PHASE3B = ROOT / "experiments" / "phase3b"


def main():
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("PathSnap Phase 3B: HMM/Viterbi Route Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Route-level evaluation comparing classifier Top-1 predictions vs HMM/Viterbi "
        "decoded predictions against supplied reference routes. All metrics computed on "
        "the 10 student-testing trips. Weak-label agreement is not verified ground-truth "
        "accuracy.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    results_file = PHASE3B / "route_evaluation.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = []

    if results:
        story.append(Paragraph("Summary Statistics", styles["Heading2"]))
        total_obs = sum(r["n_observations"] for r in results)
        total_decoded = sum(r["n_decoded"] for r in results)
        total_gaps = sum(r["route_disconnected_gaps"] for r in results)
        total_repeats = sum(r["route_repeated_traversals"] for r in results)
        total_time = sum(r["decode_time_s"] for r in results)
        avg_decode_rate = sum(r["decode_rate"] for r in results) / len(results)

        table_data = [
            ["Metric", "Value"],
            ["Trips evaluated", str(len(results))],
            ["Total observations", f"{total_obs:,}"],
            ["Total decoded", f"{total_decoded:,}"],
            ["Mean decode rate", format(avg_decode_rate, ".1%")],
            ["Disconnected gaps", str(total_gaps)],
            ["Repeated arc traversals", str(total_repeats)],
            ["Total decode time (s)", f"{total_time:.2f}"],
            ["Mean decode time/trip (s)", f"{total_time/len(results):.2f}"],
        ]

        t = Table(table_data, colWidths=[3*inch, 2*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Per-Trip Results", styles["Heading2"]))
        trip_data = [["Trip ID", "Observations", "Decoded", "Decode Rate", "Route Segments", "Route Gaps", "Decode Time (s)"]]
        for r in sorted(results, key=lambda x: x["trip_id"]):
            trip_data.append([
                r["trip_id"],
                str(r["n_observations"]),
                str(r["n_decoded"]),
                format(r["decode_rate"], ".1%"),
                str(r["route_n_segments"]),
                str(r["route_disconnected_gaps"]),
                format(r["decode_time_s"], ".2f"),
            ])
        t2 = Table(trip_data, colWidths=[1*inch, 0.9*inch, 0.8*inch, 0.9*inch, 1.1*inch, 0.8*inch, 1.0*inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t2)
        story.append(Spacer(1, 12))

    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(Paragraph(
        "Each GPS observation is matched to road-segment candidates using a trained "
        "Histogram Gradient Boosting classifier. The classifier outputs are used as "
        "emission probabilities in a Hidden Markov Model, with transitions derived from "
        "directed road-network connectivity, GPS displacement, elapsed time, implied "
        "speed, and heading agreement. Viterbi decoding finds the most likely sequence "
        "of road segments. The decoded sequence is then reconstructed into a connected "
        "route by filling gaps between consecutive decoded arcs with directed shortest "
        "paths through the road network.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Route-level metrics compare the predicted arc sequence against the supplied "
        "reference route: precision (fraction of predicted arcs in reference), recall "
        "(fraction of reference arcs in prediction), F1-score, directed connectivity "
        "(fraction of consecutive observation pairs on directly connected arcs), and "
        "Levenshtein edit distance between arc sequences.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "All metrics measure agreement with supplied reference routes, not verified "
        "ground-truth accuracy. Reference routes are used for evaluation only and are "
        "never accessed during inference.", styles["BodyText"]))

    out_path = ROOT / "docs" / "phase3b_route_report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            title="PathSnap Phase 3B Route Evaluation Report")
    doc.build(story)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
