"""python -m pathsnap.report_models — generate comparison charts and PDF."""
from __future__ import annotations

import sys

from pathsnap.cli.report_models import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
