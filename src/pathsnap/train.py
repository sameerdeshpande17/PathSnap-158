"""python -m pathsnap.train --model MODEL_NAME [--cv] [--skip-existing] [--force] [--resume RUN_ID]"""
from __future__ import annotations

import sys

from pathsnap.cli.train import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
