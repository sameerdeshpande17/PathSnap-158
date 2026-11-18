"""python -m pathsnap.decode_route --trip TRIP_ID --model MODEL"""
from __future__ import annotations

import sys

from pathsnap.cli.decode_route import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
