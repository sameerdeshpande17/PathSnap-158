"""Produce actual offline Folium browser views using installed Chromium."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import requests

from .explore import ROOT, save_json


def main():
    maps = json.loads((ROOT / "experiments/representative_maps.json").read_text())
    assets = ROOT / "visualizations/assets"
    assets.mkdir(exist_ok=True)
    records = []
    for entry in maps:
        stem = ROOT / "visualizations" / f"{entry['split']}_{entry['trip_id']}"
        html = stem.with_suffix(".html").read_text()
        urls = re.findall(r'(?:src|href)="(https://[^"]+)"', html)
        for url in urls:
            suffix = ".js" if ".js" in url.split("/")[-1] else ".css"
            name = hashlib.sha256(url.encode()).hexdigest()[:16] + suffix
            target = assets / name
            if not target.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                target.write_bytes(response.content)
            html = html.replace(url, "assets/" + name)
            records.append(dict(url=url, file=str(target.relative_to(ROOT)),
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        # Remove only raster tile initialization in the offline copy. All
        # supplied vector layers remain real Folium/Leaflet output.
        html = re.sub(r'var tile_layer_\w+ = L\.tileLayer\(.*?\);', '', html, flags=re.S)
        html = re.sub(r'tile_layer_\w+\.addTo\([^;]+;', '', html)
        html = re.sub(r'"OSM basemap \(internet required\)"\s*:\s*tile_layer_\w+,?', '', html)
        html = html.replace("Grey: roads.", "Offline vector map; no basemap tiles. Grey: roads.")
        offline = stem.with_name(stem.name + "_offline").with_suffix(".html")
        offline.write_text(html)
        screenshot = stem.with_name(stem.name + "_folium").with_suffix(".png")
        result = subprocess.run([
            "chromium", "--headless", "--disable-gpu",
            "--user-data-dir=/tmp/opencode/pathsnap-chromium",
            "--allow-file-access-from-files", "--no-first-run",
            "--hide-scrollbars", "--window-size=1400,950",
            "--virtual-time-budget=12000", f"--screenshot={screenshot}",
            offline.as_uri(),
        ], capture_output=True, text=True, timeout=180)
        (ROOT / "experiments" / f"chromium_{entry['trip_id']}.log").write_text(
            result.stdout + result.stderr)
        result.check_returncode()
        assert screenshot.exists() and screenshot.stat().st_size > 10000
    save_json(ROOT / "experiments/map_assets.json",
              list({r["url"]: r for r in records}.values()))


if __name__ == "__main__":
    main()
