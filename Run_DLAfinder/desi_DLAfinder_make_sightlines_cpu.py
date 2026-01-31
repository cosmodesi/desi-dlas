"""CPU-only sightline generation helper for unified runner inputs."""
import argparse
import os
import sys
import time


def parse_args(options=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate DESI sightlines on CPU (mock/data).",
    )
    parser.add_argument("--data-type", choices=["mock", "data"], required=True,
                        help="Dataset type.")
    parser.add_argument("--spectra-root", type=str, required=True,
                        help="Root directory of spectra files.")
    parser.add_argument("--sightline-root", type=str, required=True,
                        help="Root directory to store sightlines.")
    parser.add_argument("--list-cache-root", type=str, required=True,
                        help="Directory for cached file lists.")
    parser.add_argument("--scratch-out", type=str, default=None,
                        help="Optional output root for predictions/catalogs (cache helper).")
    parser.add_argument("--release", type=str, default="",
                        help="Release label (used for list cache name).")
    parser.add_argument("--survey", type=str, default="",
                        help="Survey label (used for list cache name).")
    parser.add_argument("--program", type=str, default="",
                        help="Program label (used for list cache name).")
    parser.add_argument("--version", type=str, default="",
                        help="Version label (used for list cache name).")
    parser.add_argument("--list-cache-tag", type=str, default="",
                        help="Override list cache tag name.")
    parser.add_argument("--rebuild-list", action="store_true",
                        help="Rebuild file list cache even if it exists.")
    parser.add_argument("--output-layout", choices=["k/j", "k"], default=None,
                        help="Output directory layout relative to sightline root.")
    parser.add_argument("--spectra-pattern", type=str, default="",
                        help="Spectra filename pattern (use {id}).")
    parser.add_argument("--zbest-pattern", type=str, default="",
                        help="ZBEST filename pattern (use {id}).")
    parser.add_argument("--truth-pattern", type=str, default="",
                        help="Truth filename pattern (use {id}); empty to skip.")
    parser.add_argument("--sightline-pattern", type=str, default="",
                        help="Sightline filename pattern (use {id}).")
    parser.add_argument("--value", type=int, default=0,
                        help="Start index in the file list.")
    parser.add_argument("--length", type=int, default=None,
                        help="Number of files to process (default: run to end).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of CPU workers (0/1 = serial).")
    parser.add_argument("--force-sightlines", action="store_true",
                        help="Regenerate sightlines even if they exist.")
    if options is None:
        return parser.parse_args()
    return parser.parse_args(options)


def _build_task(item):
    spectra_path, truth_path, zbest_path, sightline_path, data_type, force = item
    if not spectra_path:
        return None
    if os.path.exists(sightline_path) and not force:
        return ("skip", sightline_path)
    os.makedirs(os.path.dirname(sightline_path), exist_ok=True)
    if data_type == "mock":
        truth_arg = truth_path if truth_path and os.path.exists(truth_path) else []
    else:
        truth_arg = []
    zbest_arg = zbest_path if zbest_path and os.path.exists(zbest_path) else []
    try:
        from desidlas.datasets.get_sightlines import get_sightlines
        get_sightlines(spectra_path, truth_arg, zbest_arg, sightline_path)
    except Exception as exc:
        return ("fail", spectra_path, str(exc))
    return ("ok", sightline_path)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["DESIDLAS_FORCE_CPU"] = "1"

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    args = parse_args()

    from Run_DLAfinder.desi_DLAfinder_run import _default_patterns
    from Run_DLAfinder.desi_DLAfinder_run import _load_or_build_lists

    defaults = _default_patterns(args.data_type)
    patterns = {
        "output_layout": args.output_layout or defaults["output_layout"],
        "spectra_pattern": args.spectra_pattern or defaults["spectra_pattern"],
        "zbest_pattern": args.zbest_pattern or defaults["zbest_pattern"],
        "truth_pattern": args.truth_pattern or defaults["truth_pattern"],
        "sightline_pattern": args.sightline_pattern or defaults["sightline_pattern"],
        "pred_pattern": defaults["pred_pattern"],
        "dlacat_pattern": defaults["dlacat_pattern"],
    }

    list_start = time.time()
    lists = _load_or_build_lists(args, patterns)
    print(f"File list size: {len(lists['sightline'])} (loaded in {time.time() - list_start:.2f}s)")

    start = args.value
    end = len(lists["sightline"]) if args.length is None else min(start + args.length, len(lists["sightline"]))
    if start >= end:
        print(f"No work for range {start}:{end}.")
        return

    tasks = []
    for spectra_path, truth_path, zbest_path, sightline_path in zip(
        lists["spectra"][start:end],
        lists["truth"][start:end],
        lists["zbest"][start:end],
        lists["sightline"][start:end],
    ):
        tasks.append((spectra_path, truth_path, zbest_path, sightline_path, args.data_type, args.force_sightlines))

    start_time = time.time()
    if args.workers and args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(_build_task, tasks, chunksize=1)
            for result in results:
                if not result:
                    continue
                if result[0] == "fail":
                    print(f"Failed to build sightlines for {result[1]}: {result[2]}")
    else:
        for item in tasks:
            result = _build_task(item)
            if result and result[0] == "fail":
                print(f"Failed to build sightlines for {result[1]}: {result[2]}")

    print(f"Sightline generation completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
