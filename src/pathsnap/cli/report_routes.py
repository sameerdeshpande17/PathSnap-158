"""python -m pathsnap.report_routes

Generate route-level report (charts + PDF) from saved route evaluation outputs.

Does NOT train or evaluate models. Reads existing artifacts under
artifacts/models/evaluation/ and experiments/phase3b/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.units import inch

from pathsnap.explore import ROOT

PHASE3B = ROOT / "experiments" / "phase3b"
EVAL_DIR = ROOT / "artifacts" / "models" / "evaluation"
VIS = ROOT / "visualizations" / "phase3b"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.report_routes",
        description="Generate route-level comparison report from saved outputs.",
    )
    p.add_argument(
        "--output", default="docs/phase3b_route_report.pdf",
        help="PDF output path (default: docs/phase3b_route_report.pdf)",
    )
    p.add_argument(
        "--no-pdf", action="store_true",
        help="Generate charts only, skip PDF",
    )
    return p.parse_args(argv)


def _load_route_eval_results() -> pd.DataFrame:
    rows = []
    if not PHASE3B.exists():
        return pd.DataFrame(rows)
    for f in sorted(PHASE3B.glob("*_route_eval_*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def _bar_chart(df, metric, title, fname):
    if df.empty or metric not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    means = df.groupby("model")[metric].mean()
    x = np.arange(len(means))
    ax.bar(x, means.values)
    ax.set_xticks(x, [m.replace("_", " ").title() for m in means.index])
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(VIS / fname, dpi=150)
    plt.close(fig)
    return VIS / fname


def main(argv=None) -> int:
    args = parse_args(argv)
    VIS.mkdir(parents=True, exist_ok=True)
    df = _load_route_eval_results()
    charts = []
    for metric, title, fname in [
        ("classifier_hmm_agreement_rate", "Classifier-HMM Agreement Rate", "route_agreement.png"),
        ("route_n_disconnected_gaps", "Route Disconnected Gaps (per trip)", "route_gaps.png"),
        ("route_n_repeated_traversals", "Route Repeated Arc Traversals", "route_repeats.png"),
        ("timing_s", "Decode Time per Trip (s)", "route_time.png"),
    ]:
        c = _bar_chart(df, metric, title, fname)
        if c is not None:
            charts.append(c)
    print(f"[pathsnap.report_routes] {len(charts)} charts -> {VIS}", flush=True)

    if args.no_pdf:
        print("[pathsnap.report_routes] charts only (--no-pdf)", flush=True)
        return 0

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as exc:
        print(f"[pathsnap.report_routes] reportlab unavailable: {exc}; charts only",
              flush=True)
        return 0

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("PathSnap Phase 3B: Route-Level Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Generated from saved route evaluation outputs. Weak-label agreement is not "
        "verified ground-truth accuracy. No training was performed during report "
        "generation.", styles["BodyText"]))
    story.append(Spacer(1, 10))
    for c in charts:
        story.append(Image(str(c), width=6.5 * inch, height=2.6 * inch))
        story.append(Spacer(1, 8))
    if not df.empty:
        story.append(Paragraph("Summary table", styles["Heading2"]))
        summary = df.groupby("model")[
            ["classifier_hmm_agreement_rate", "route_n_disconnected_gaps",
             "route_n_repeated_traversals", "timing_s"]
        ].mean().reset_index()
        html = summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")
        story.append(Paragraph(html, styles["BodyText"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from reportlab.lib.units import inch
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            title="PathSnap Phase 3B Route Report")
    doc.build(story)
    print(f"[pathsnap.report_routes] PDF -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
