from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from futureflu import PipelineConfig, SeasonConfig, run_pipeline


# Construct the theta range from user-supplied config values.
# 根据配置内容构建 theta 参数范围。
def _build_theta_range(config: Dict) -> List[float]:
    if "theta_range" in config:
        return [float(v) for v in config["theta_range"]]

    start = float(config.get("theta_start", 0.2))
    end = float(config.get("theta_end", 0.51))
    step = float(config.get("theta_step", 0.01))

    theta_values = []
    current = start
    while current < end - 1e-9:
        theta_values.append(round(current, 2))
        current += step
    return theta_values


# Load a pipeline configuration from a JSON file on disk.
# 从磁盘 JSON 文件加载流水线配置。
def load_config(path: Path) -> PipelineConfig:
    data = json.loads(path.read_text(encoding="utf-8"))

    seasons: List[SeasonConfig] = []
    for entry in data["seasons"]:
        years = entry.get("years")
        if years is None:
            years = [entry["year"]]
        for year in years:
            seasons.append(
                SeasonConfig(
                    year=int(year),
                    hemisphere=str(entry["hemisphere"]),
                    epi_csv=Path(entry["epi_csv"]).resolve(),
                )
            )

    theta_range = _build_theta_range(data)

    aggregate_path = (
        Path(data["aggregate_fitness_path"]).resolve()
        if data.get("aggregate_fitness_path")
        else None
    )
    trimmed_path = (
        Path(data["trimmed_fitness_path"]).resolve()
        if data.get("trimmed_fitness_path")
        else None
    )

    seq_processes = data.get("sequence_processes")
    if seq_processes is not None:
        seq_processes = int(seq_processes)

    return PipelineConfig(
        subtype=str(data["subtype"]),
        fasta_path=Path(data["fasta_path"]).resolve(),
        info_path=Path(data["info_path"]).resolve(),
        evescape_prefix=str(data["evescape_prefix"]),
        evescape_dir=Path(data["evescape_dir"]).resolve(),
        sequence_cutoff=str(data.get("sequence_cutoff", "2024-02-01")),
        theta_range=theta_range,
        seasons=seasons,
        output_root=Path(data.get("output_root", "./pipeline_outputs")).resolve(),
        aggregate_fitness_path=aggregate_path,
        trimmed_fitness_path=trimmed_path,
        sequence_processes=seq_processes,
        lineage_column=str(data.get("lineage_column", "clade")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated FutureFlu pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to pipeline JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    generated = run_pipeline(config)
    for key, value in generated.items():
        if isinstance(value, pd.DataFrame):
            print(f"{key}: dataframe shape={value.shape}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
