"""python -m pathsnap <command> — PathSnap ML pipeline dispatcher."""
from __future__ import annotations

import sys

from pathsnap.cli import evaluate, predict, prepare_features, report_models, train
from pathsnap.cli.compare_tuned import main as compare_tuned_main
from pathsnap.cli.decode_route import main as decode_route_main
from pathsnap.cli.evaluate_routes import main as evaluate_routes_main
from pathsnap.cli.optimize import main as optimize_main
from pathsnap.cli.report_routes import main as report_routes_main
from pathsnap.cli.select_final_model import main as select_final_model_main
from pathsnap.cli.refit_final import main as refit_final_main
from pathsnap import audit_evaluation

COMMANDS = {
    "audit_evaluation": audit_evaluation.main,
    "compare_tuned": compare_tuned_main,
    "decode_route": decode_route_main,
    "evaluate": evaluate.main,
    "evaluate_routes": evaluate_routes_main,
    "optimize": optimize_main,
    "prepare_features": prepare_features.main,
    "predict": predict.main,
    "refit_final": refit_final_main,
    "report_models": report_models.main,
    "report_routes": report_routes_main,
    "select_final_model": select_final_model_main,
    "train": train.main,
}


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m pathsnap <command> [options]")
        print("")
        print("Commands:")
        for name in sorted(COMMANDS):
            print(f"  {name}")
        print("")
        print("Run 'python -m pathsnap <command> --help' for command options.")
        return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(COMMANDS))}", file=sys.stderr)
        return 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
