"""python -m pathsnap.evaluate_routes --split student_test --model MODEL"""
from __future__ import annotations

import sys

from pathsnap.cli.evaluate_routes import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
