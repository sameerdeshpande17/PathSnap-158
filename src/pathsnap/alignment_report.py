"""Generate measured Phase 2A summaries, comparison maps and PDF report."""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, LongTable, PageBreak

from .alignment import geometry
from .alignment_run import OUT, RAW, LABELS
from .data import load_trip
from .explore import ROOT, save_json

VIS = ROOT / "visualizations/alignment"


def aggregate():
    comparison = pd.read_csv(ROOT / "experiments/alignment_comparison.csv", dtype={"trip_id": str})
    differences = pd.read_csv(OUT / "method_disagreements.csv", dtype={"trip_id": str})
    summaries, gaps, excluded = [], [], []
    for (split, method), group in comparison.groupby(["split", "method"]):
        frames = []
        for row in group.itertuples():
            frame = pd.read_csv(LABELS / split / f"{row.trip_id}_{method}.csv",
                                dtype={"trip_id": str})
            # First observations have no observed sampling interval.
            frame["dt_s"] = frame.timestamp.diff()
            frames.append(frame)
        all_rows = pd.concat(frames, ignore_index=True)
        residuals = all_rows.projection_distance_m.to_numpy()
        row = dict(
            split=split, method=method, trips=len(group), observations=len(all_rows),
            assigned_fraction=float(all_rows.assigned_arc_id.notna().mean()),
            quality_eligible_fraction=float(all_rows.quality_eligible.mean()),
            distance_median_m=float(np.median(residuals)),
            distance_p95_m=float(np.quantile(residuals, .95)),
            residual_rmse_m=float(np.sqrt(np.mean(residuals ** 2))),
            route_order_violations=int(group.route_order_violations.sum()),
            unexplained_jumps=int(group.unexplained_jumps.sum()),
            backward_progress_steps=int(group.backward_progress_steps.sum()),
            reference_break_crossings=int(group.reference_break_crossings.sum()),
            occurrence_coverage_macro=float(group.occurrence_coverage_fraction.mean()),
            sequence_monotone_subsequence_macro=float(group.sequence_monotone_subsequence_fraction.mean()),
            runtime_s=float(group.runtime_s.sum()),
            consensus_screened_count=int(all_rows.recommended_supervision_use.sum()),
            consensus_screened_fraction=float(all_rows.recommended_supervision_use.mean()),
        )
        summaries.append(row)
        for label, flag in [
            ("dt<=1s", all_rows.dt_s <= 1),
            ("1<dt<=5s", (all_rows.dt_s > 1) & (all_rows.dt_s <= 5)),
            ("5<dt<=30s", (all_rows.dt_s > 5) & (all_rows.dt_s <= 30)),
            ("dt>30s", all_rows.dt_s > 30),
        ]:
            d = all_rows.loc[flag, "projection_distance_m"]
            gaps.append(dict(split=split, method=method, interval=label,
                             observations=len(d),
                             distance_median_m=float(d.median()) if len(d) else None,
                             distance_p95_m=float(d.quantile(.95)) if len(d) else None))
        flags = Counter(flag for value in all_rows.quality_flags.dropna()
                        for flag in value.split(";") if flag)
        excluded.extend(dict(split=split, method=method, flag=key, observations=value)
                        for key, value in sorted(flags.items()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(OUT / "gap_sensitivity.csv", index=False)
    pd.DataFrame(excluded).to_csv(OUT / "quality_flag_counts.csv", index=False)
    disagreement_summary = []
    for (split, first, second), group in differences.groupby(["split", "first", "second"]):
        disagreement_summary.append(dict(
            split=split, pair=f"{first}/{second}",
            arc_disagreement_fraction=float(np.average(group.arc_disagreement_fraction,
                                                       weights=group.observations)),
            occurrence_disagreement_fraction=float(np.average(
                group.occurrence_disagreement_fraction, weights=group.observations)),
        ))
    pd.DataFrame(disagreement_summary).to_csv(OUT / "disagreement_summary.csv", index=False)
    save_json(LABELS / "methods.json", json.loads((OUT / "frozen_methods.json").read_text()))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("quality_eligible_fraction", "Quality-eligible fraction"),
        ("distance_p95_m", "Pooled spatial residual p95 (m)"),
        ("route_order_violations", "Route-order violations"),
        ("unexplained_jumps", "Unexplained progress jumps"),
        ("occurrence_coverage_macro", "Mean occurrence coverage"),
        ("runtime_s", "Alignment runtime (s)"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for index, split in enumerate(["train", "student_test"]):
            values = summary[summary.split == split].set_index("method").loc[["A", "B", "C"], key]
            ax.bar(np.arange(3) + index * .35, values, width=.35, label=split)
        ax.set_xticks(np.arange(3) + .175, ["A", "B", "C"])
        ax.set_title(title)
        if key in ["distance_p95_m", "runtime_s"]:
            ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Reference-assisted alignment diagnostics — not pointwise accuracy")
    fig.tight_layout()
    fig.savefig(VIS / "alignment_comparison.png", dpi=150)
    plt.close(fig)
    gap_frame = pd.DataFrame(gaps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["dt<=1s", "1<dt<=5s", "5<dt<=30s", "dt>30s"]
    for ax, split in zip(axes, ["train", "student_test"]):
        for method in ["A", "B", "C"]:
            group = gap_frame[(gap_frame.split == split) & (gap_frame.method == method)]
            ax.plot(labels, group.set_index("interval").loc[labels].distance_p95_m,
                    marker="o", label=method)
        ax.set_title(split)
        ax.set_ylabel("Spatial residual p95 (m)")
        ax.tick_params(axis="x", rotation=15)
        ax.legend()
    fig.tight_layout()
    fig.savefig(VIS / "timestamp_gap_sensitivity.png", dpi=150)
    plt.close(fig)
    return comparison, summary, differences


def make_map(split, trip_id, reason):
    trip = load_trip(RAW / split / trip_id)
    g = geometry(trip)
    frames = {m: pd.read_csv(LABELS / split / f"{trip_id}_{m}.csv") for m in ["A", "B", "C"]}
    occurrences = np.array([frames[m].assigned_occurrence_index.to_numpy() for m in frames])
    disagreement = np.any(occurrences != occurrences[0], axis=0)
    lon, lat = trip.track[:, :2].mean(axis=0)
    inverse = Transformer.from_crs(
        CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"),
        "EPSG:4326", always_xy=True)
    m = folium.Map([lat, lon], tiles=None, prefer_canvas=True)
    reference = folium.FeatureGroup(name="Full reference route — green", show=True)
    folium.PolyLine(trip.nodes[trip.arcs[trip.route]][:, :, ::-1].tolist(),
                    color="green", weight=6, opacity=.6,
                    tooltip="Full reference arcs; no invented connectors").add_to(reference)
    reference.add_to(m)
    gps = folium.FeatureGroup(name="Raw GPS — blue", show=True)
    for i, (longitude, latitude, timestamp) in enumerate(trip.track):
        folium.CircleMarker([latitude, longitude], radius=1, weight=.5,
                            color="blue", fill=True,
                            tooltip=f"GPS {i}, t={timestamp:g}s").add_to(gps)
    gps.add_to(m)
    for method, color, radius in [("A", "orange", 2), ("B", "purple", 1.5), ("C", "red", 1)]:
        frame = frames[method]
        path = frame.assigned_occurrence_index.to_numpy()
        fraction = np.divide(frame.route_progress_m.to_numpy() - g.starts[path],
                             g.lengths[path], out=np.zeros(len(path)), where=g.lengths[path] > 0)
        arc = trip.arcs[trip.route[path]]
        xy = g.nodes_xy[arc[:, 0]] + fraction[:, None] * (
            g.nodes_xy[arc[:, 1]] - g.nodes_xy[arc[:, 0]])
        projected_lon, projected_lat = inverse.transform(xy[:, 0], xy[:, 1])
        group = folium.FeatureGroup(name=f"{method} aligned projections — {color}", show=True)
        for i, (x, y) in enumerate(zip(projected_lon, projected_lat)):
            folium.CircleMarker([y, x], radius=radius, weight=.6, opacity=.65,
                                color=color, fill=False,
                                tooltip=f"{method}: GPS {i}, occurrence {path[i]}, "
                                f"arc {trip.route[path[i]]}, d={frame.projection_distance_m.iloc[i]:.2f}m"
                                ).add_to(group)
        group.add_to(m)
    group = folium.FeatureGroup(name="Occurrence disagreements — yellow", show=True)
    for i in np.flatnonzero(disagreement):
        folium.CircleMarker(trip.track[i, :2][::-1].tolist(), radius=3, weight=.4,
                            color="#b59b00", opacity=.35, fill=False,
                            tooltip=f"GPS {i}; A/B/C occurrences {occurrences[:, i].tolist()}").add_to(group)
    group.add_to(m)
    extent = np.vstack([trip.track[:, :2], trip.nodes[trip.arcs[trip.route].ravel()]])
    m.fit_bounds([extent.min(axis=0)[::-1].tolist(), extent.max(axis=0)[::-1].tolist()])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:8px;font:13px sans-serif">'
        f'{split} {trip_id}: {reason}<br>Blue GPS; green full reference; '
        'A orange / B purple / C red projections; yellow disagreements.<br>'
        'Weak alignment only. Offline vector map; toggle overlapping layers.</div>'))
    stem = VIS / f"{split}_{trip_id}"
    m.save(str(stem.with_suffix(".html")))
    html = stem.with_suffix(".html").read_text()
    assets = json.loads((ROOT / "experiments/map_assets.json").read_text())
    for asset in assets:
        html = html.replace(asset["url"], "../assets/" + Path(asset["file"]).name)
    # No remote dependencies should remain in script/style links.
    assert not re.findall(r'(?:src|href)="https://', html)
    stem.with_suffix(".html").write_text(html)
    process = subprocess.run([
        "chromium", "--headless", "--disable-gpu", "--no-first-run",
        "--user-data-dir=/tmp/opencode/pathsnap-alignment-chromium",
        "--allow-file-access-from-files", "--hide-scrollbars",
        "--window-size=1400,950", "--virtual-time-budget=20000",
        f"--screenshot={stem.with_suffix('.png')}", stem.with_suffix(".html").as_uri(),
    ], capture_output=True, text=True, timeout=180)
    (OUT / f"map_{split}_{trip_id}.log").write_text(process.stdout + process.stderr)
    process.check_returncode()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for method, color in [("A", "orange"), ("B", "purple"), ("C", "red")]:
        axes[0].plot(frames[method].assigned_occurrence_index, color=color, label=method, alpha=.7)
        axes[1].plot(frames[method].projection_distance_m, color=color, label=method, alpha=.7)
    for i in np.flatnonzero(g.dt > 5):
        axes[0].axvline(i, color="grey", alpha=.2)
    axes[0].set_ylabel("Reference occurrence index")
    axes[0].set_title(f"{split} {trip_id}: {reason}; grey vertical lines = dt>5s")
    axes[1].set(xlabel="GPS observation index", ylabel="Spatial residual (m)")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(stem.with_name(stem.name + "_sequence").with_suffix(".png"), dpi=150)
    plt.close(fig)
    return dict(split=split, trip_id=trip_id, reason=reason,
                occurrence_disagreements=int(disagreement.sum()),
                html=str(stem.with_suffix(".html").relative_to(ROOT)))


def investigation(summary, comparison):
    breaks = pd.read_csv(ROOT / "experiments/reference_adjacency_breaks.csv", dtype={"trip_id": str})
    repeated = json.loads((OUT / "repeated_arc_examples.json").read_text())
    stats = pd.read_csv(ROOT / "experiments/trip_statistics.csv", dtype={"trip_id": str})
    lines = [
        "# Reference-route alignment investigation", "",
        "## Official meaning and missing alignment rules",
        "Dataset Exploration.pdf section 3 defines `.route` as a chronological sequence",
        "of arc IDs traversed by the vehicle; arc IDs and node IDs are zero-based row",
        "indices. Sections 3 and 6 explicitly warn against direct row correspondence",
        "and treating supplied reference information as error-free physical truth.",
        "Models and Training Guidelines.pdf section 5 asks for corresponding reference",
        "segments for Top-1 scoring, but provides no timestamp-to-occurrence alignment",
        "algorithm. Searches of all five official text extractions found no explicit",
        "alignment rule or per-observation label file. Approximate alignment is therefore",
        "an independently implemented weak-label policy, not recovered official labels.", "",
        "## Ordered directed traversal and repeated occurrences",
        f"Across 90 trips, no route-row count equals GPS-row count. There are "
        f"{int(stats.route_count.sum()):,} supplied arc occurrences.",
        f"{len(repeated)} trips contain repeated arc IDs. Repeated occurrences are consistent",
        "with repeated traversal in the documented format; geometry alone cannot prove",
        "that each repeat reflects actual vehicle motion rather than a reference error.",
        "Do not deduplicate the route. A nearest-segment tie cannot locate the visit in time.",
    ]
    for item in repeated:
        lines.append(f"- {item['split']} {item['trip_id']}: arc {item['arc_id']} occurs at "
                     f"zero-based positions {item['occurrences']}.")
    lines.extend([
        "", "## All 14 directed adjacency breaks",
        f"{int(breaks.directed_path_exists.sum())} boundaries have a directed connector in "
        "the supplied road graph; that connector is not represented between these",
        "consecutive route occurrences. This is consistent with skipped connector arcs,",
        "partial route encoding or reference errors, but does not identify the true driven path.",
        "All 14 endpoint pairs are in the same weak component. None is explained merely",
        "by separate weak components or coincident node coordinates.",
        "Trip 00000086 has no forward directed connector at occurrences 1->2, but does",
        "have a reverse connector: a direction/connectivity inconsistency in the supplied",
        "reference/network pair. The data alone cannot establish which source is wrong.",
        "Shortest paths are evidence of graph possibilities, not silent route repairs.",
        "Endpoint-to-nearest-GPS indices/times in the CSV are spatial diagnostics,",
        "not adjudicated temporal alignment (especially on revisited roads).", "",
        "| Split/trip | Occurrences | Arcs | End/start nodes | Separation m | Directed path m |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in breaks.itertuples():
        path_length = f"{row.directed_path_m:.3f}" if row.directed_path_exists else "unreachable"
        lines.append(f"| {row.split}/{row.trip_id} | {row.occurrence_before}->{row.occurrence_after} "
                     f"| {row.arc_before}->{row.arc_after} | {row.before_to}->{row.after_from} "
                     f"| {row.separation_m:.3f} | {path_length} |")
    lines.extend([
        "", "## Timestamp and geometry evidence",
        "All observed timestamps strictly increase. Coordinates and trajectory order allow",
        "distance-based matching and nondecreasing reference-occurrence alignment, with",
        "multiple observations per arc and skipped arcs. They do not uniquely identify",
        "road direction at overlapping geometries or revisit number at repeated arcs.",
        "Trip 00000090 contains a 153064-second gap; the vehicle's route during that interval",
        "cannot be verified from these observations. Speed bounds and progression penalties",
        "are assumptions and are relaxed with elapsed time, not treated as new observations.",
        "", "## Executed methods, screening and conclusions",
        "All three methods were run on all 80 training and 10 student-testing trips.",
        "See docs/alignment_methodology.md for equations and limitations; frozen settings",
        "and training-only calibration provenance are under experiments/alignment/.",
        "Only 100% tentative assignment coverage is guaranteed by the algorithms; eligibility",
        "and conservative screened coverage must be examined separately.",
    ])
    for row in summary.itertuples():
        lines.append(f"- {row.split} {row.method}: residual median {row.distance_median_m:.3f}m, "
                     f"p95 {row.distance_p95_m:.3f}m; quality eligible "
                     f"{row.quality_eligible_fraction:.2%}; order violations "
                     f"{row.route_order_violations}; unexplained jumps {row.unexplained_jumps}.")
    sensitivity = pd.read_csv(OUT / "training_sensitivity.csv")
    for method in ["B", "C"]:
        values = sensitivity[(sensitivity.method == method) & (sensitivity.variant != "wide")]
        lines.append(f"- {method}: maximum benchmark arc disagreement under sigma x0.5/x2 "
                     f"was {values.arc_disagreement_fraction.max():.2%}.")
    failures = pd.read_csv(OUT / "training_failure_cases.csv", dtype={"trip_id": str})
    lines.extend(["", "## Important observed failure of method C",
                  "The probabilistic method is not robust enough for unscreened labels.",
                  "On the following training trips it locks onto an early occurrence",
                  "and remains there as the GPS trajectory progresses. Hard progression",
                  "constraints combined with beam/candidate truncation and no unmatched",
                  "state can prevent recovery. These are measured alignment failures,",
                  "not evidence of incorrect GPS labels or verified correctness of B."])
    for row in failures.itertuples():
        lines.append(
            f"- train {row.trip_id}: C terminal occurrence {row.terminal_C_occurrence}, "
            f"B terminal {row.terminal_B_occurrence}, reference size "
            f"{row.total_reference_occurrences}; C p95 {row.C_residual_p95_m:.2f}m "
            f"versus B {row.B_residual_p95_m:.2f}m. "
            f"{row.observations_C_over_1000m_B_under_100m} observations have C distance "
            f">1km while B distance <100m; first at observation "
            f"{row.first_severe_observation}. Selected for supervised use among these: "
            f"{row.severe_observations_selected_for_supervision}. During the constant "
            f"suffix, the next occurrence was speed-admissible in "
            f"{row.subsequent_occurrence_speed_admissible_steps} steps, but in the "
            f"nearest candidate set only "
            f"{row.subsequent_occurrence_in_nearest_candidates_steps} steps.")
    lines.extend([
        "The screened-consensus rule excludes these severe residuals. No algorithm or",
        "parameter was changed after student testing to conceal the failures. Future",
        "training-only work could test explicit unmatched states, gap-aware re-entry,",
        "soft outlier transition mixtures, and broader lookahead. This version does",
        "not claim those remedies were implemented or validated.",
    ])
    lines.extend([
        "", "Approach B is the better-supported default chronology heuristic for continued",
        "label investigation: it respects occurrence order, avoids C's observed lock-in",
        "failures, and was less sensitive to the tested spatial-scale perturbations.",
        "This is a robustness judgement,",
        "not a pointwise-accuracy ranking. A provides the spatial-residual lower bound;",
        "C adds a constrained probabilistic alternative but does not establish calibrated",
        "confidence or independent correctness.",
        "For initial supervised experiments, use only the screened B/C occurrence-consensus",
        "subset (stored as recommended_supervision_use in C exports), retaining all",
        "uncertain rows for audit. B and C targets are identical on that subset.",
        "These labels are suitable only for explicitly weak-supervised exploratory training.",
        "They are not adequate to claim true pointwise accuracy or F1 on future classifiers.",
        "No model was trained. Independent point labels or human adjudication remain necessary",
        "to validate the target itself. The separate Phase 1 package remains unlocated.",
    ])
    (ROOT / "docs/route_alignment_investigation.md").write_text("\n".join(lines) + "\n")


def report(summary, maps):
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    story = []
    def text(value, style="BodyText"):
        story.append(Paragraph(escape(str(value)), styles[style]))
        story.append(Spacer(1, 6))
    def document(path):
        buffer = []
        for line in path.read_text().splitlines() + [""]:
            if line.startswith("#") or not line.strip():
                if buffer:
                    text(" ".join(buffer))
                    buffer.clear()
                if line.startswith("#"):
                    text(line.lstrip("# "), "Heading2")
            elif line.startswith("|"):
                continue  # A proper PDF table below presents these break rows.
            else:
                buffer.append(line.strip())
    def table(rows, widths):
        tab = LongTable([[Paragraph(escape(str(c)), styles["BodyText"]) for c in r]
                         for r in rows], colWidths=widths, repeatRows=1)
        tab.setStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                      ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
                      ("VALIGN", (0, 0), (-1, -1), "TOP")])
        story.extend([tab, Spacer(1, 10)])
    def image(path, caption):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            width, height = im.size
        story.append(Image(str(path), width=500, height=500 * height / width))
        text(caption)
    text("PathSnap Phase 2A: Reference Alignment and Weak-Label Validation", "Title")
    text("Locally executed evidence. No independent pointwise ground truth; no classifier trained. "
         "AI-assisted implementation and reporting; team review and understanding are required.")
    text("1. Measured comparison", "Heading1")
    table([["Split / method", "Eligible %", "Median / p95 m", "Order violations", "Jump flags", "Runtime s"]] +
          [[f"{r.split} / {r.method}", f"{100*r.quality_eligible_fraction:.2f}",
            f"{r.distance_median_m:.2f} / {r.distance_p95_m:.2f}",
            r.route_order_violations, r.unexplained_jumps, f"{r.runtime_s:.3f}"]
           for r in summary.itertuples()], [115, 60, 105, 75, 65, 80])
    text("All methods tentatively assign every observation. Eligibility excludes poor residuals "
         "and temporal/topological flags; it is not verified correctness.")
    for r in summary[summary.method == "C"].itertuples():
        text(f"{r.split}: screened B/C consensus = {r.consensus_screened_count:,} of "
             f"{r.observations:,} observations ({r.consensus_screened_fraction:.2%}).")
    image(VIS / "alignment_comparison.png", "Figure 1. Actual partition-separated diagnostics.")
    image(VIS / "timestamp_gap_sensitivity.png", "Figure 2. Residual sensitivity to observed timestamp intervals.")
    disagreement = pd.read_csv(OUT / "disagreement_summary.csv")
    text("Method disagreement (not independent error)", "Heading2")
    table([["Split", "Pair", "Arc disagreement %", "Occurrence disagreement %"]] +
          [[r.split, r.pair, f"{100*r.arc_disagreement_fraction:.3f}",
            f"{100*r.occurrence_disagreement_fraction:.3f}"] for r in disagreement.itertuples()],
          [130, 70, 150, 150])
    gaps = pd.read_csv(OUT / "gap_sensitivity.csv")
    text("Observed timestamp-gap strata", "Heading2")
    table([["Split/method", "Interval", "Observations", "Residual p95 m"]] +
          [[f"{r.split}/{r.method}", r.interval, r.observations,
            f"{r.distance_p95_m:.3f}" if np.isfinite(r.distance_p95_m) else "N/A"]
           for r in gaps.itertuples()], [150, 100, 110, 140])
    text("2. Investigation", "Heading1")
    document(ROOT / "docs/route_alignment_investigation.md")
    breaks = pd.read_csv(ROOT / "experiments/reference_adjacency_breaks.csv", dtype={"trip_id": str})
    table([["Trip", "Occurrence boundary", "Endpoint separation m", "Directed connector m"]] +
          [[r.trip_id, f"{r.occurrence_before}->{r.occurrence_after}", f"{r.separation_m:.3f}",
            f"{r.directed_path_m:.3f}" if r.directed_path_exists else "unreachable"]
           for r in breaks.itertuples()], [110, 120, 135, 135])
    text("3. Methodology and limitations", "Heading1")
    document(ROOT / "docs/alignment_methodology.md")
    frozen = json.loads((OUT / "frozen_methods.json").read_text())
    text("Frozen parameters", "Heading2")
    table([["Parameter", "Value"]] + list(frozen["config"].items()), [250, 250])
    sensitivity = pd.read_csv(OUT / "training_sensitivity.csv", dtype={"trip_id": str})
    text("4. Training-only sensitivity benchmarks", "Heading1")
    table([["Trip", "Method", "Variant", "Arc disagreement %", "Occurrence disagreement %"]] +
          [[r.trip_id, r.method, r.variant, f"{100*r.arc_disagreement_fraction:.3f}",
            f"{100*r.occurrence_disagreement_fraction:.3f}"] for r in sensitivity.itertuples()],
          [100, 50, 120, 115, 115])
    text("Disagreement is relative to the base heuristic, not error against independent truth. "
         "No assignments changed under the wider search on these training examples.")
    text("5. Real geographical and sequence visualisations", "Heading1")
    for entry in maps:
        story.append(PageBreak())
        stem = VIS / f"{entry['split']}_{entry['trip_id']}"
        text(f"{entry['split']} {entry['trip_id']}: {entry['reason']}", "Heading2")
        image(stem.with_suffix(".png"), f"Actual Folium browser render; interactive map: {entry['html']}. "
              "Projection markers overlap where methods agree; controls allow separate inspection.")
        image(stem.with_name(stem.name + "_sequence").with_suffix(".png"),
              "Occurrence chronology and spatial residuals from saved alignments.")
    text("6. Verification, runtime and reproduction", "Heading1")
    text((OUT / "tests.log").read_text())
    run = json.loads((OUT / "run.json").read_text())
    text(f"All-trip run elapsed wall time: {run['elapsed_s']:.3f}s; "
         f"trip-method runs: {run['trip_method_runs']}; models trained: {run['models_trained']}.")
    text("Frozen configuration hash: " + run["config_sha256_before_and_after"])
    text("Commands: PYTHONPATH=src ~/ml312/bin/python -m pathsnap.alignment_run "
         "{fit,benchmark,freeze,run}, in that order; then alignment_breaks, alignment_report "
         "and alignment_verify. See README for exact shell commands and logs.")
    text("Outputs: experiments/alignment_comparison.csv, reference_adjacency_breaks.csv, "
         "experiments/alignment/ summaries and provenance; data/alignments/train and "
         "student_test per-trip/method CSVs; visualizations/alignment HTML and PNG files.")
    text("Conclusion: B has better observed parameter robustness, but no method's true "
         "pointwise correctness is established. Screened B/C-consensus labels support "
         "exploratory weak supervision only. Independent target validation remains open.")
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 22, f"PathSnap Phase 2A | weak supervision, not ground truth | page {doc.page}")
    SimpleDocTemplate(str(ROOT / "docs/phase2a_alignment_report.pdf"),
                      pagesize=(612, 792), leftMargin=36, rightMargin=36,
                      topMargin=36, bottomMargin=36,
                      title="PathSnap Phase 2A Alignment Report").build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-maps", action="store_true",
                        help="Rebuild tables/report using already generated map screenshots")
    args = parser.parse_args()
    VIS.mkdir(parents=True, exist_ok=True)
    comparison, summary, differences = aggregate()
    training_bc = differences[(differences.split == "train") & (differences.first == "B") &
                              (differences.second == "C")]
    worst = training_bc.loc[training_bc.occurrence_disagreement_fraction.idxmax(), "trip_id"]
    selections = [
        ("train", worst, "Largest training B/C occurrence disagreement"),
        ("train", "00000090", "Largest training timestamp gap"),
        ("train", "00000069", "Repeated training route occurrences"),
        ("student_test", "00000056", "Held-out repeated traversal; frozen parameters"),
    ]
    if args.reuse_maps:
        maps = json.loads((OUT / "comparison_maps.json").read_text())
    else:
        maps = []
        seen = set()
        for split, trip_id, reason in selections:
            if (split, trip_id) not in seen:
                maps.append(make_map(split, trip_id, reason))
                seen.add((split, trip_id))
                print(f"MAP {split}/{trip_id}", flush=True)
        save_json(OUT / "comparison_maps.json", maps)
    investigation(summary, comparison)
    report(summary, maps)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
