#!/usr/bin/env python3
"""Build the 2013-2024 fitness tables for all subtypes.
为所有亚型生成 2013-2024 年度适应度表。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futureflu.pipeline import run_pipeline
from run_pipeline import load_config

DEFAULT_CONFIGS = [
    "configs/h3n2_pre2024.json",
    "configs/h1n1_pre2024.json",
    "configs/victoria_pre2024.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fitness tables for all subtypes.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="List of pipeline config files (default: pre-2024 trio).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a consolidated fitness table (disabled by default).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="Parallel jobs (default: 6).",
    )
    return parser.parse_args()


def _run_single(args: Tuple[Path, Optional[int], Optional[str]]) -> dict:
    cfg_file, seq_processes, hemi_filter = args
    config = load_config(cfg_file)
    config.aggregate_fitness_path = None
    config.trimmed_fitness_path = None
    config.sequence_processes = seq_processes
    if hemi_filter is not None:
        filtered_seasons = [
            season for season in config.seasons if season.hemisphere.lower() == hemi_filter
        ]
        if not filtered_seasons:
            return {"data": pd.DataFrame(columns=["subtype", "hemisphere", "year", "rank", "risk_mutation_group", "mutation_count", "fitness", "clade"])}
        config.seasons = filtered_seasons
    return run_pipeline(config)


def main() -> None:
    args = parse_args()
    config_paths = [Path(p).resolve() for p in args.configs]
    for cfg_file in config_paths:
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

    seq_processes = 1 if args.jobs and args.jobs > 1 else None
    tasks: List[Tuple[Path, Optional[int], Optional[str]]] = []
    for cfg in config_paths:
        config_preview = load_config(cfg)
        hemispheres = sorted({season.hemisphere.lower() for season in config_preview.seasons})
        for hemi in hemispheres:
            tasks.append((cfg, seq_processes, hemi))

    if args.jobs <= 1:
        results = [_run_single(task) for task in tasks]
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as executor:
            future_to_task = {executor.submit(_run_single, task): task for task in tasks}
            results = []
            for future in as_completed(future_to_task):
                results.append(future.result())

    frames = [res["data"] for res in results]

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["subtype", "hemisphere", "year", "rank"])
        .reset_index(drop=True)
    )

    trimmed_columns = [
        "subtype",
        "hemisphere",
        "year",
        "rank",
        "risk_mutation_group",
        "mutation_count",
        "fitness",
        "clade",
    ]
    trimmed = combined[trimmed_columns]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trimmed.to_csv(args.output, index=False)
        print(f"fitness table written to {args.output} ({trimmed.shape[0]} rows)")


if __name__ == "__main__":
    main()
