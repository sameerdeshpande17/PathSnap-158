"""python -m pathsnap.refit_final --model MODEL_NAME

Explicitly refit the selected model on the full training partition and save
the final checkpoint to artifacts/models/final/<model>/.
"""
from __future__ import annotations

import sys

from pathsnap.cli.refit_final import main

if __name__ == "__main__":
    sys.exit(main())
