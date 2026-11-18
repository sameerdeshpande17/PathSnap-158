"""Independent all-output checks and preservation audit for Phase 2A."""
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import shapely
from PIL import Image

from .alignment_run import OUT, RAW, LABELS, digest
from .data import load_trip, project
from .explore import ROOT, save_json


def main():
    calibration = json.loads((OUT / "calibration.json").read_text())
    expected_train = {p.name for p in (RAW / "train").iterdir() if p.is_dir()}
    expected_test = {p.name for p in (RAW / "student_test").iterdir() if p.is_dir()}
    assert len(expected_train) == 80 and len(expected_test) == 10
    assert not (expected_train & expected_test)
    assert calibration["fit_split"] == "train"
    assert {p["trip_id"] for p in calibration["provenance"]} == expected_train
    frozen_path = OUT / "frozen_methods.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["calibration_sha256"] == digest(OUT / "calibration.json")
    assert frozen["sensitivity_sha256"] == digest(OUT / "training_sensitivity.csv")
    run = json.loads((OUT / "run.json").read_text())
    assert run["config_sha256_before_and_after"] == digest(frozen_path)
    checked_files, checked_rows, max_error = 0, 0, 0.
    for split, ids in [("train", expected_train), ("student_test", expected_test)]:
        assert len(list((LABELS / split).glob("*.csv"))) == len(ids) * 3
        for trip_id in sorted(ids):
            trip = load_trip(RAW / split / trip_id)
            gps, nodes = project(trip.track, trip.nodes)
            frames = {}
            for method in ["A", "B", "C"]:
                path = LABELS / split / f"{trip_id}_{method}.csv"
                assert path.stat().st_mtime_ns >= frozen_path.stat().st_mtime_ns
                frame = pd.read_csv(path, dtype={"trip_id": str})
                frames[method] = frame
                assert len(frame) == len(trip.track)
                assert (frame.trip_id == trip_id).all() and (frame.split == split).all()
                assert (frame.alignment_method == method).all()
                np.testing.assert_array_equal(frame.observation_index, np.arange(len(frame)))
                np.testing.assert_allclose(frame[["longitude", "latitude", "timestamp"]], trip.track,
                                           rtol=0, atol=1e-12)
                occ = frame.assigned_occurrence_index.to_numpy()
                assert np.issubdtype(occ.dtype, np.integer)
                assert ((occ >= 0) & (occ < len(trip.route))).all()
                np.testing.assert_array_equal(frame.assigned_arc_id, trip.route[occ])
                if method in ["B", "C"]:
                    assert (np.diff(occ) >= 0).all()
                independent_distance = shapely.distance(
                    shapely.points(gps), shapely.linestrings(nodes[trip.arcs[trip.route[occ]]]))
                error = float(np.max(np.abs(independent_distance - frame.projection_distance_m)))
                max_error = max(max_error, error)
                np.testing.assert_allclose(independent_distance, frame.projection_distance_m, atol=1e-6)
                assert frame.spatial_compatibility_score.between(0, 1).all()
                assert (frame.label_status == "weak_inferred").all()
                checked_files += 1
                checked_rows += len(frame)
            recommended = frames["C"].recommended_supervision_use.to_numpy()
            b, c = frames["B"], frames["C"]
            expected = ((b.assigned_occurrence_index == c.assigned_occurrence_index) &
                        b.quality_eligible & c.quality_eligible &
                        ~c.quality_flags.fillna("").str.contains("geometric_occurrence_ambiguity"))
            np.testing.assert_array_equal(recommended, expected)
            assert not frames["A"].recommended_supervision_use.any()
            assert not frames["B"].recommended_supervision_use.any()
    preservation = []
    for item in json.loads((ROOT / "experiments/deliverable_manifest.json").read_text()):
        if item["path"] == "README.md":
            continue  # Explicitly requested reproduction-command update.
        path = ROOT / item["path"]
        assert path.exists(), path
        assert digest(path) == item["sha256"], f"Phase 1 changed: {path}"
        preservation.append(item["path"])
    save_json(OUT / "phase1_preservation.json", {
        "verified_unchanged_files": len(preservation),
        "exception": "README.md extended with requested Phase 2A instructions",
        "paths": preservation,
    })
    test_log = (OUT / "tests.log").read_text()
    assert "Ran 19 tests" in test_log and test_log.strip().endswith("OK")
    maps = json.loads((OUT / "comparison_maps.json").read_text())
    for entry in maps:
        html = ROOT / entry["html"]
        assert html.exists()
        with Image.open(html.with_suffix(".png")) as image:
            image.verify()
    pdf = ROOT / "docs/phase2a_alignment_report.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    (OUT / "pdfinfo.txt").write_text(info.stdout)
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                          text=True, check=True).stdout
    assert "Reference" in text and "weak" in text.lower() and "00000086" in text
    (OUT / "report_extracted.txt").write_text(text)
    save_json(OUT / "verification.json", {
        "tests_passed": 19, "alignment_files_verified": checked_files,
        "label_rows_verified": checked_rows,
        "independent_shapely_distance_max_error_m": max_error,
        "train_only_fit_provenance_verified": True,
        "frozen_config_hash_and_export_order_verified": True,
        "phase1_files_preserved": len(preservation),
        "maps_verified": len(maps), "pdf_text_verified": True,
        "models_trained": 0,
    })
    output_paths = [
        ROOT / "docs/phase2a_alignment_report.pdf",
        ROOT / "docs/route_alignment_investigation.md",
        ROOT / "docs/alignment_methodology.md",
        ROOT / "experiments/alignment_comparison.csv",
        ROOT / "experiments/reference_adjacency_breaks.csv",
        ROOT / "README.md",
    ]
    for folder in [OUT, LABELS, ROOT / "visualizations/alignment"]:
        output_paths.extend(p for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json")
    output_paths.extend((ROOT / "src/pathsnap").glob("alignment*.py"))
    output_paths.append(ROOT / "tests/test_alignment.py")
    save_json(OUT / "manifest.json", [
        dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(set(output_paths))])
    print((OUT / "verification.json").read_text())


if __name__ == "__main__":
    main()
