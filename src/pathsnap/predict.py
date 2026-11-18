"""python -m pathsnap.predict --model MODEL_NAME --trip TRIP_ID [--split SPLIT] [--run-id RUN_ID]"""
from __future__ import annotations

import sys

from pathsnap.cli.predict import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
