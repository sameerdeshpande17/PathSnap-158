"""Repeatable, hash-checked extraction of the two supplied archives."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / "data/original/manifest.json").read_text())
    for item in manifest:
        archive = ROOT / "data/original" / item["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"], archive
        dest = ROOT / ("docs/official" if item["archive"] ==
                       "PathSnap_student_materials.zip" else "data/raw")
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == item["members"]
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                contents = z.read(member)
                if target.exists():
                    if target.read_bytes() != contents:
                        raise ValueError(f"Refusing to overwrite modified input: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
    for pdf in sorted((ROOT / "docs/official").glob("*.pdf")):
        subprocess.run(["pdftotext", "-layout", str(pdf),
                        str(pdf.with_suffix(".txt"))], check=True)
    print("Both archive hashes verified; extraction complete.")


if __name__ == "__main__":
    main()
