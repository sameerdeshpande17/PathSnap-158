"""python -m pathsnap.optimize --model MODEL_NAME --trials N"""
from __future__ import annotations

import sys

from pathsnap.cli.optimize import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
