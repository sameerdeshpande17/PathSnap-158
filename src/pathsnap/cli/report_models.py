"""python -m pathsnap.report_models

Generate model comparison charts and a PDF report from completed experiment
outputs. No training is performed here.
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

from pathsnap.explore import ROOT

OUT = ROOT / "experiments" / "phase2b"
ART = ROOT / "artifacts" / "models"
VIS = ROOT / "visualizations" / "phase2b"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.report_models",
        description="Generate model comparison report from saved outputs.")
    p.add_argument("--output", default="docs/phase2b_model_report.pdf",
                   help="PDF output path (default: docs/phase2b_model_report.pdf)")
    p.add_argument("--no-pdf", action="store_true",
                   help="Generate charts only, skip PDF")
    return p.parse_args(argv)


def _load_eval_results() -> pd.DataFrame:
    rows = []
    ev_dir = ART / "evaluation"
    if not ev_dir.exists():
        return pd.DataFrame(rows)
    for f in sorted(ev_dir.glob("*.json")):
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
    stds = df.groupby("model")[metric].std()
    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=stds.values.fillna(0), capsize=3)
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
    df = _load_eval_results()
    charts = []
    for metric, title, fname in [
        ("top1_agreement", "Student-test Top-1 weak-label agreement", "report_top1.png"),
        ("candidate_macro_f1", "Candidate Macro F1", "report_macrof1.png"),
        ("candidate_miss_rate", "Candidate miss rate", "report_miss_rate.png"),
        ("mrr", "Mean reciprocal rank", "report_mrr.png"),
        ("eval_seconds", "Evaluation seconds", "report_eval_time.png"),
    ]:
        c = _bar_chart(df, metric, title, fname)
        if c is not None:
            charts.append(c)
    print(f"[pathsnap.report_models] {len(charts)} charts written to {VIS}",
          flush=True)

    if args.no_pdf:
        print("[pathsnap.report_models] charts only (--no-pdf)", flush=True)
        return 0

    # Minimal PDF via reportlab if available
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as exc:
        print(f"[pathsnap.report_models] reportlab unavailable: {exc}; "
              f"charts generated only", flush=True)
        return 0

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("PathSnap Phase 2B Model Comparison Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Generated from saved evaluation outputs. Weak-label agreement is not "
        "verified ground-truth accuracy. No training was performed during report "
        "generation.", styles["BodyText"]))
    story.append(Spacer(1, 10))
    for c in charts:
        story.append(Image(str(c), width=6.5 * inch, height=2.6 * inch))
        story.append(Spacer(1, 8))
    if not df.empty:
        story.append(Paragraph("Summary table", styles["Heading2"]))
        summary = df.groupby("model")[
            ["top1_agreement", "candidate_macro_f1", "candidate_miss_rate",
             "mrr", "eval_seconds"]].mean().reset_index()
        html = summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")
        story.append(Paragraph(html, styles["BodyText"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            title="PathSnap Phase 2B Model Report")
    doc.build(story)
    print(f"[pathsnap.report_models] PDF written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
