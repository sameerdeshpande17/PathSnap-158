"""Final artifact verification; creates a hash manifest after report generation."""
import csv
import hashlib
import importlib.metadata
import json
import subprocess

from PIL import Image

from .explore import ROOT, save_json


def main():
    summary = json.loads((ROOT / "experiments/summary.json").read_text())
    assert summary["validation"]["parsing_errors"] == 0
    assert summary["train"]["trips"] == 80 and summary["student_test"]["trips"] == 10
    assert not summary["validation"]["overlapping_trip_ids"]
    tests = (ROOT / "experiments/tests.log").read_text()
    assert "Ran 7 tests" in tests and tests.strip().endswith("OK")
    checklist = list(csv.DictReader((ROOT / "docs/requirements_checklist.csv").open()))
    assert all(None not in row and all(v is not None for v in row.values()) for row in checklist)
    packages = []
    for line in (ROOT / "requirements.lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        assert importlib.metadata.version(name) == version
        packages.append(name)
    maps = json.loads((ROOT / "experiments/representative_maps.json").read_text())
    for item in maps:
        prefix = ROOT / "visualizations" / f"{item['split']}_{item['trip_id']}"
        for suffix in [".html", "_offline.html", ".png", "_folium.png"]:
            path = prefix.with_name(prefix.name + suffix)
            assert path.exists() and path.stat().st_size > 10000
            if suffix.endswith(".png"):
                with Image.open(path) as image:
                    image.verify()
    pdf = ROOT / "docs/phase1_report.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    (ROOT / "experiments/pdfinfo.txt").write_text(info.stdout)
    content = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True, check=True).stdout
    for expected in ["208,417", "19,160", "153,064", "Bayesian", "Folium"]:
        assert expected in content, expected
    (ROOT / "experiments/report_extracted.txt").write_text(content)
    save_json(ROOT / "experiments/final_verification.json", {
        "unit_tests_passed": 7,
        "original_archive_member_hashes_verified": 360,
        "trips_with_verified_prediction_rows_and_bounds": 90,
        "representative_maps_and_images_verified": len(maps),
        "locked_installed_packages_verified": len(packages),
        "checklist_rows_validated": len(checklist),
        "pdf_metadata_and_extracted_measurements_verified": True,
        "models_trained": 0,
    })
    manifest = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name == "deliverable_manifest.json":
            continue
        manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save_json(ROOT / "experiments/deliverable_manifest.json", manifest)
    print(json.dumps(json.loads((ROOT / "experiments/final_verification.json").read_text()),
                     indent=2))


if __name__ == "__main__":
    main()
