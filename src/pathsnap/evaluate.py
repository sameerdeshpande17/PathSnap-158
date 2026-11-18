"""python -m pathsnap.evaluate --model MODEL_NAME [--split SPLIT] [--run-id RUN_ID]"""
from __future__ import annotations

import sys

from pathsnap.cli.evaluate import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
