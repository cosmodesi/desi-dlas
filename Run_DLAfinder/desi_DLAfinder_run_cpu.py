"""CPU-only wrapper for the unified DESI DLA finder."""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Run_DLAfinder import desi_DLAfinder_run


def _parse_cpu_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-workers", type=int, default=None,
                        help="Number of CPU worker processes for prediction.")
    parser.add_argument("--no-force-cpu", action="store_true",
                        help="Do not set DESIDLAS_FORCE_CPU.")
    return parser.parse_known_args(argv)


def main():
    cpu_args, rest = _parse_cpu_args(sys.argv[1:])
    if not cpu_args.no_force_cpu:
        os.environ["DESIDLAS_FORCE_CPU"] = "1"
    if cpu_args.cpu_workers is not None:
        os.environ["DESIDLAS_CPU_WORKERS"] = str(cpu_args.cpu_workers)

    # Ensure --cpu-only is always passed to the unified runner.
    if "--cpu-only" not in rest:
        rest.append("--cpu-only")

    sys.argv = [sys.argv[0]] + rest
    desi_DLAfinder_run.main()


if __name__ == "__main__":
    main()
