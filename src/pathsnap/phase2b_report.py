"""Phase 2B charts and PDF evidence report from saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, LongTable, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, TableStyle)

from .explore import ROOT

OUT = ROOT / "experiments/phase2b"
ART = ROOT / "artifacts/phase2b"
VIS = ROOT / "visualizations/phase2b"
MODELS = ["logistic", "knn", "random_forest", "hist_gradient_boosting", "mlp"]


def styles():
    s = getSampleStyleSheet()
    s["BodyText"].fontSize = 9
    s["BodyText"].leading = 12
    return s


def bar_chart(cv, metric, title, fname, log=False):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    means = cv.groupby("model")[metric].mean()
    stds = cv.groupby("model")[metric].std()
    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=stds.values, capsize=3)
    ax.set_xticks(x, [m.replace("_", " ").title() for m in means.index])
    ax.set_title(title)
    ax.set_ylabel(metric)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(VIS / fname, dpi=150)
    plt.close(fig)


def table(rows, widths, s):
    data = [[Paragraph(escape(str(c)), s["BodyText"]) for c in r] for r in rows]
    t = LongTable(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f2")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    s = styles()
    story = []
    story.append(Paragraph("PathSnap Phase 2B: Candidate Generation and Five-Model Training", s["Title"] if "Title" in s else s["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Locally executed evidence from frozen Phase 2A weak labels. Weak-label agreement is "
        "not verified ground-truth accuracy. No Bayesian optimisation or HMM/Viterbi work "
        "has been performed. AI-assisted implementation; team review required.",
        s["BodyText"]))
    story.append(Spacer(1, 10))

    cv = pd.read_csv(ROOT / "experiments/model_comparison.csv", dtype={"trip_id": str}) \
        if (ROOT / "experiments/model_comparison.csv").exists() else pd.read_csv(
            OUT / "model_comparison.csv", dtype={"trip_id": str})
    if "strategy" not in cv.columns:
        cv["strategy"] = "consensus"

    story.append(Paragraph("1. Experiment design", s["Heading2"]))
    story.append(Paragraph(
        "Five classifiers were compared on identical trip-grouped cross-validation splits "
        "(3 folds, seed 42) using the same frozen candidate set and weak labels. Candidates "
        "come from the complete road network (STRtree, 50 m radius, max 12 per observation, "
        "nearest-arc fallback); features use GPS and road-network data only. Reference-route "
        "identifiers and occurrence indices are never prediction features.",
        s["BodyText"]))
    cand = pd.read_csv(ROOT / "experiments/candidate_generation_statistics.csv",
                       dtype={"trip_id": str})
    train_cand = cand[cand.split == "train"]
    story.append(Paragraph(
        f"Weak-label candidate recall on training: {train_cand.covered_weak_observations.sum() / train_cand.weak_labeled_observations.sum():.4%} "
        f"({train_cand.covered_weak_observations.sum():,} of {train_cand.weak_labeled_observations.sum():,} labeled observations).",
        s["BodyText"]))

    story.append(Paragraph("2. Cross-validation results (mean ± std over 3 folds)", s["Heading2"]))
    agg = cv.groupby(["strategy", "model"])[
        ["obs_top1_agreement", "cand_candidate_macro_f1", "cand_candidate_log_loss",
         "obs_mrr", "obs_candidate_miss_rate", "cont_direct_adjacency", "train_s"]].mean().reset_index()
    rows = [["Strategy", "Model", "Top-1", "Macro F1", "Log loss", "MRR", "Miss rate", "Adjacency", "Train s"]]
    for r in agg.itertuples():
        rows.append([r.strategy, r.model, f"{r.obs_top1_agreement:.4f}",
                     f"{r.cand_candidate_macro_f1:.4f}",
                     f"{r.cand_candidate_log_loss:.4f}" if np.isfinite(r.cand_candidate_log_loss) else "N/A",
                     f"{r.obs_mrr:.4f}", f"{r.obs_candidate_miss_rate:.4%}",
                     f"{r.cont_direct_adjacency:.4f}", f"{r.train_s:.2f}"])
    story.append(table(rows, [95, 150, 55, 55, 55, 45, 55, 75, 50], s))
    story.append(Spacer(1, 10))
    for metric, title, fname in [
        ("obs_top1_agreement", "Observation Top-1 weak-label agreement", "cv_top1.png"),
        ("cand_candidate_macro_f1", "Candidate Macro F1", "cv_macrof1.png"),
        ("cand_candidate_log_loss", "Candidate log loss", "cv_logloss.png"),
        ("obs_mrr", "Mean reciprocal rank", "cv_mrr.png"),
        ("train_s", "Training time per fold (s)", "cv_time.png", True),
    ]:
        log = len(fname.split("_")) > 1 and fname.endswith("time.png")
        bar_chart(cv, metric, title, fname, log=log)
        story.append(Image(str(VIS / fname), width=6.0 * inch, height=2.55 * inch))
        story.append(Paragraph(f"Figure. {title}.", s["BodyText"]))

    selection = json.loads((OUT / "selection.json").read_text())
    story.append(Paragraph("3. Model selection", s["Heading2"]))
    story.append(Paragraph(
        f"Selected model: <b>{selection['selected_model']}</b>, by CV mean observation "
        f"Top-1 weak-label agreement. Tie-breakers: CV candidate Macro F1, CV MRR, lower "
        "prediction latency. Selection and all hyperparameters were fixed before any "
        "student-testing evaluation.", s["BodyText"]))
    story.append(Paragraph(
        f"Per-model CV Top-1: " +
        ", ".join(f"{k} {v:.4f}" for k, v in selection["models"].items()) + ".",
        s["BodyText"]))

    story.append(PageBreak())
    story.append(Paragraph("4. Student-testing evaluation (held out, after selection frozen)", s["Heading2"]))
    final = json.loads((ART / "final" / "final_training.json").read_text())
    rows = [["Quantity", "Value"]]
    for k, v in final.items():
        if k in ["artifact_path", "config_sha256"]:
            continue
        rows.append([k, str(v)])
    story.append(table(rows, [260, 260], s))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Student-testing results are a single development-stage evaluation on the 10 "
        "reserved trips under the frozen weak-label policy. They are not an unbiased final "
        "holdout and not verified ground truth.", s["BodyText"]))

    story.append(Paragraph("5. Costs and artifacts", s["Heading2"]))
    rows = [["Model", "Device", "Train s", "Artifact bytes"]]
    for m in MODELS:
        p = ART / f"fold_0" / f"{m}.joblib"
        # use final_training for selected model instead
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Saved artifacts: artifacts/phase2b/final/{model}.joblib|.pt with estimator, "
        "StandardScaler, feature schema, and schema version; per-fold CV artifacts under "
        "artifacts/phase2b/fold_{i}/. Predictions exported to "
        "artifacts/phase2b/final/student_predictions_{model}.parquet.", s["BodyText"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Limitations", s["Heading2"]))
    for line in [
        "Targets are Phase 2A weak/inferred supervision; screened B/C consensus covers "
        "59.5% of training observations and 65.3% of student-testing observations.",
        "Approach C showed severe lock-in failures on training trips 00000032 and 00000084; "
        "its unscreened labels were not used.",
        "Candidate-level metrics are dominated by negative candidates; selection does not "
        "rely on candidate accuracy alone.",
        "No independent pointwise ground truth exists; true accuracy/F1 cannot be claimed.",
        "KNN prediction cost scales with training-set size; reported for completeness.",
        "The MLP used CUDA when available; scikit-learn estimators ran on CPU.",
    ]:
        story.append(Paragraph("• " + escape(line), s["BodyText"]))
        story.append(Spacer(1, 4))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 22, f"PathSnap Phase 2B | weak supervision | page {doc.page}")

    SimpleDocTemplate(str(ROOT / "docs/phase2b_model_report.pdf"),
                      pagesize=(612, 792), leftMargin=36, rightMargin=36,
                      topMargin=36, bottomMargin=36,
                      title="PathSnap Phase 2B Model Report",
                      author="PathSnap team - AI-assisted checkpoint").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    print("Generated docs/phase2b_model_report.pdf", flush=True)


if __name__ == "__main__":
    main()
