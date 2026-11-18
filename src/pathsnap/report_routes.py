"""python -m pathsnap.report_routes"""
from __future__ import annotations

import sys

from pathsnap.cli.report_routes import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
