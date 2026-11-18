"""python -m pathsnap.prepare_features — generate and cache candidate features."""
from __future__ import annotations

import sys

from pathsnap.cli.prepare_features import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
