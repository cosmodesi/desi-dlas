"""Unified DESI DLA finder for mock/data (sightline generation + prediction + catalog)."""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np


def parse_args(options=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unified DESI DLA finder for mock/data."
    )

    parser.add_argument("--data-type", choices=["mock", "data"], required=True,
                        help="Dataset type.")
    parser.add_argument("--spectra-root", type=str, default=None,
                        help="Root directory of spectra files (required to generate sightlines).")
    parser.add_argument("--sightline-root", type=str, required=True,
                        help="Root directory to store sightlines.")
    parser.add_argument("--list-cache-root", type=str, required=True,
                        help="Directory for cached file lists.")

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
    parser.add_argument("--pred-pattern", type=str, default="",
                        help="Prediction filename pattern (use {id}).")
    parser.add_argument("--dlacat-pattern", type=str, default="",
                        help="DLA catalog filename pattern (use {id}).")

    parser.add_argument("--value", type=int, default=0,
                        help="Start index in the file list.")
    parser.add_argument("--length", type=int, default=None,
                        help="Number of files to process (default: run to end).")

    parser.add_argument("--batch-size", type=int, default=512,
                        help="Sightlines per batch (GPU).")
    parser.add_argument("--max-windows", type=int, default=16384,
                        help="Windows per GPU batch.")
    parser.add_argument("--scratch-out", type=str, default=None,
                        help="Optional output root for prediction/catalog files.")

    parser.add_argument("--generate-sightlines", action="store_true",
                        help="Generate sightlines if missing.")
    parser.add_argument("--force-sightlines", action="store_true",
                        help="Regenerate sightlines even if they exist.")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Disable GPU for prediction.")
    parser.add_argument("--stack-dlacat", action="store_true",
                        help="Stack per-file DLA catalogs into one FITS.")
    parser.add_argument("--stack-output", type=str, default=None,
                        help="Output FITS path for stacked catalog.")
    parser.add_argument("--stack-scope", choices=["range", "all"], default="all",
                        help="Stack catalogs from current range or all cached files.")
    parser.add_argument("--skip-existing-pred", action="store_true",
                        help="Skip prediction/catalog generation when pred output already exists.")
    parser.add_argument("--fill-missing-dlacat", action="store_true",
                        help="When skipping predictions, fill missing catalogs from existing pred files.")

    if options is None:
        return parser.parse_args()
    return parser.parse_args(options)


def _default_patterns(data_type: str):
    if data_type == "mock":
        return {
            "output_layout": "k/j",
            "spectra_pattern": "spectra-16-{id}.fits",
            "zbest_pattern": "zbest-16-{id}.fits",
            "truth_pattern": "",
            "sightline_pattern": "sightlines-{id}.npy",
            "pred_pattern": "sightlines-pred_gpu-{id}.npy",
            "dlacat_pattern": "dlacat_gpu-{id}.fits",
        }
    return {
        "output_layout": "k",
        "spectra_pattern": "spectra-main-dark-{id}.fits.gz",
        "zbest_pattern": "zbest-16-{id}.fits",
        "truth_pattern": "",
        "sightline_pattern": "{id}-pre-sightlines.npy",
        "pred_pattern": "{id}-pre-sightlines-pred.npy",
        "dlacat_pattern": "{id}-dlacat.fits",
    }


def _make_output_path(root, layout, group, leaf, filename):
    if layout == "k/j":
        out_dir = os.path.join(root, group, leaf)
    else:
        out_dir = os.path.join(root, group)
    return os.path.join(out_dir, filename)


def _build_cache_tag(args):
    if args.list_cache_tag:
        return args.list_cache_tag
    parts = [args.data_type]
    for value in (args.release, args.survey, args.program, args.version):
        if value:
            parts.append(value)
    if not parts:
        return "default"
    return "_".join(parts)


def _discover_files(args, patterns):
    spectra_root = args.spectra_root
    if not spectra_root:
        raise ValueError("spectra_root is required to build file list.")

    spectra_list = []
    truth_list = []
    zbest_list = []
    sightline_list = []
    pred_list = []
    dlacat_list = []
    group_list = []
    leaf_list = []

    for group in sorted(os.listdir(spectra_root)):
        group_path = os.path.join(spectra_root, group)
        if not os.path.isdir(group_path):
            continue
        for leaf in sorted(os.listdir(group_path)):
            leaf_path = os.path.join(group_path, leaf)
            if not os.path.isdir(leaf_path):
                continue

            spectra_name = patterns["spectra_pattern"].format(id=leaf)
            spectra_path = os.path.join(leaf_path, spectra_name)
            if not os.path.exists(spectra_path):
                continue

            zbest_name = patterns["zbest_pattern"].format(id=leaf) if patterns["zbest_pattern"] else ""
            zbest_path = os.path.join(leaf_path, zbest_name) if zbest_name else ""

            truth_name = patterns["truth_pattern"].format(id=leaf) if patterns["truth_pattern"] else ""
            truth_path = os.path.join(leaf_path, truth_name) if truth_name else ""

            sightline_name = patterns["sightline_pattern"].format(id=leaf)
            sightline_path = _make_output_path(
                args.sightline_root, patterns["output_layout"], group, leaf, sightline_name
            )

            pred_root = args.scratch_out or args.sightline_root
            pred_name = patterns["pred_pattern"].format(id=leaf)
            pred_path = _make_output_path(
                pred_root, patterns["output_layout"], group, leaf, pred_name
            )

            dlacat_name = patterns["dlacat_pattern"].format(id=leaf)
            dlacat_path = _make_output_path(
                pred_root, patterns["output_layout"], group, leaf, dlacat_name
            )

            spectra_list.append(spectra_path)
            truth_list.append(truth_path)
            zbest_list.append(zbest_path)
            sightline_list.append(sightline_path)
            pred_list.append(pred_path)
            dlacat_list.append(dlacat_path)
            group_list.append(group)
            leaf_list.append(leaf)

    return {
        "spectra": np.array(spectra_list, dtype=object),
        "truth": np.array(truth_list, dtype=object),
        "zbest": np.array(zbest_list, dtype=object),
        "sightline": np.array(sightline_list, dtype=object),
        "pred": np.array(pred_list, dtype=object),
        "dlacat": np.array(dlacat_list, dtype=object),
        "group": np.array(group_list, dtype=object),
        "leaf": np.array(leaf_list, dtype=object),
    }


def _load_or_build_lists(args, patterns):
    os.makedirs(args.list_cache_root, exist_ok=True)
    cache_tag = _build_cache_tag(args)
    cache_path = os.path.join(args.list_cache_root, f"filelist_{cache_tag}.npz")

    if os.path.exists(cache_path) and not args.rebuild_list:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    data = _discover_files(args, patterns)
    np.savez(cache_path, **data)
    return data


def _ensure_parent_dirs(paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _stack_dla_catalogs(dlacat_paths, output_path):
    from astropy.table import Table, vstack

    existing = [p for p in dlacat_paths if p and os.path.exists(p)]
    if not existing:
        print("No DLA catalog files found for stacking.")
        return

    print(f"Stacking {len(existing)} catalogs into {output_path}")
    base_table = Table.read(existing[0], format="fits")
    for path in existing[1:]:
        append_table = Table.read(path, format="fits")
        base_table = vstack([base_table, append_table])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_table.write(output_path, format="fits", overwrite=True)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    args = parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    defaults = _default_patterns(args.data_type)
    patterns = {
        "output_layout": args.output_layout or defaults["output_layout"],
        "spectra_pattern": args.spectra_pattern or defaults["spectra_pattern"],
        "zbest_pattern": args.zbest_pattern or defaults["zbest_pattern"],
        "truth_pattern": args.truth_pattern or defaults["truth_pattern"],
        "sightline_pattern": args.sightline_pattern or defaults["sightline_pattern"],
        "pred_pattern": args.pred_pattern or defaults["pred_pattern"],
        "dlacat_pattern": args.dlacat_pattern or defaults["dlacat_pattern"],
    }

    print("\n" + "=" * 80)
    print("DESI DLA FINDER - UNIFIED RUNNER")
    print("=" * 80)
    print(f"Program started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data type: {args.data_type}")
    print(f"Output layout: {patterns['output_layout']}")
    print(f"Sightline root: {args.sightline_root}")
    if args.scratch_out:
        print(f"Scratch output: {args.scratch_out}")
    print("=" * 80 + "\n")

    list_start = time.time()
    lists = _load_or_build_lists(args, patterns)
    list_elapsed = time.time() - list_start
    print(f"File list size: {len(lists['sightline'])} (loaded in {list_elapsed:.2f}s)")

    start = args.value
    if args.length is None:
        end = len(lists["sightline"])
    else:
        end = min(start + args.length, len(lists["sightline"]))
    if start >= end:
        print(f"No work for range {start}:{end}.")
        return

    spectra_sel = lists["spectra"][start:end]
    truth_sel = lists["truth"][start:end]
    zbest_sel = lists["zbest"][start:end]
    sightline_sel = lists["sightline"][start:end]
    pred_sel = lists["pred"][start:end]
    dlacat_sel = lists["dlacat"][start:end]

    if args.generate_sightlines:
        from desidlas.datasets.get_sightlines import get_sightlines

        print("\nGenerating sightlines (if missing)...")
        gen_start = time.time()
        for spectra_path, truth_path, zbest_path, sightline_path in zip(
            spectra_sel, truth_sel, zbest_sel, sightline_sel
        ):
            if not spectra_path:
                continue
            if os.path.exists(sightline_path) and not args.force_sightlines:
                continue
            os.makedirs(os.path.dirname(sightline_path), exist_ok=True)
            if args.data_type == "mock":
                truth_arg = []
            else:
                truth_arg = truth_path if truth_path and os.path.exists(truth_path) else []
            zbest_arg = zbest_path if zbest_path and os.path.exists(zbest_path) else []
            try:
                get_sightlines(spectra_path, truth_arg, zbest_arg, sightline_path)
            except Exception as exc:
                print(f"Failed to build sightlines for {spectra_path}: {exc}")
        print(f"Sightline generation completed in {time.time() - gen_start:.2f}s\n")

    existing_mask = np.array([os.path.exists(p) for p in sightline_sel])
    if not np.any(existing_mask):
        print("No sightlines found for prediction.")
        return

    sightline_sel = sightline_sel[existing_mask]
    pred_sel = pred_sel[existing_mask]
    dlacat_sel = dlacat_sel[existing_mask]

    if args.skip_existing_pred:
        pred_exists = np.array([os.path.exists(p) for p in pred_sel])
        if np.all(pred_exists):
            if args.fill_missing_dlacat:
                from desidlas.prediction.pred_sightline import save_pred_all
                missing = np.array([not os.path.exists(p) for p in dlacat_sel])
                if np.any(missing):
                    print("Filling missing dlacat from existing predictions...")
                    save_pred_all(sightline_sel[missing], pred_sel[missing], dlacat_sel[missing])
                else:
                    print("All predictions and catalogs already exist; skipping.")
            else:
                print("All predictions already exist; skipping prediction.")
            return
        sightline_sel = sightline_sel[~pred_exists]
        pred_sel = pred_sel[~pred_exists]
        dlacat_sel = dlacat_sel[~pred_exists]
        if len(sightline_sel) == 0:
            print("No pending predictions after skip-existing check.")
            return

    _ensure_parent_dirs(pred_sel)
    _ensure_parent_dirs(dlacat_sel)

    if args.cpu_only:
        from desidlas.prediction.multiprocess_partprediction import predictions_desi
        predict_fn = predictions_desi
    else:
        from desidlas.prediction.multiprocess_partprediction_gpu import predictions_desi_gpu
        predict_fn = lambda s, p: predictions_desi_gpu(
            s, p, batch_size=args.batch_size, max_windows_per_batch=args.max_windows
        )

    predict_start = time.time()
    predict_fn(sightline_sel, pred_sel)
    print(f"Prediction completed in {time.time() - predict_start:.2f}s")

    from desidlas.prediction.pred_sightline import save_pred_all
    cat_start = time.time()
    save_pred_all(sightline_sel, pred_sel, dlacat_sel)
    print(f"Catalog generation completed in {time.time() - cat_start:.2f}s")

    if args.stack_dlacat:
        if args.stack_output:
            stack_out = args.stack_output
        else:
            base_root = args.scratch_out or args.sightline_root
            stack_out = os.path.join(base_root, "dlacat.fits")
        stack_list = dlacat_sel if args.stack_scope == "range" else lists["dlacat"]
        _stack_dla_catalogs(stack_list, stack_out)


if __name__ == "__main__":
    main()
