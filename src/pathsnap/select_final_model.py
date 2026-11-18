"""python -m pathsnap.select_final_model

Recommend the final model from training-only CV evidence (no retraining).
"""
from __future__ import annotations

import sys

from pathsnap.cli.select_final_model import main

if __name__ == "__main__":
    sys.exit(main())
