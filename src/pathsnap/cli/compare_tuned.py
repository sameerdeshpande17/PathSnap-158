"""python -m pathsnap.compare_tuned

Compare completed Optuna tuning studies (reads saved summaries; no training).
"""
from __future__ import annotations

import sys

from pathsnap.model_selection import compare_tuned


def main(argv=None) -> int:
    compare_tuned()
    return 0


if __name__ == "__main__":
    sys.exit(main())
