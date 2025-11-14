from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .fitness import compute_fitness_for_prediction
from .prediction import run_prediction_pipeline
from .sequence import generate_sequence_table


# Configuration for a single subtype, hemisphere, and year.
# 单个亚型、半球与年份的配置定义。
@dataclass
class SeasonConfig:
    year: int
    # Hemisphere label, either "North" or "South".
    # 半球标签，可选 "North" 或 "South"。
    hemisphere: str
    epi_csv: Path


# Top-level configuration for building fitness tables for one subtype.
# 单个亚型生成适应度表的顶层配置。
@dataclass
class PipelineConfig:
    subtype: str
    fasta_path: Path
    info_path: Path
    evescape_prefix: str
    evescape_dir: Path
    sequence_cutoff: str
    theta_range: Sequence[float]
    seasons: List[SeasonConfig]
    output_root: Path
    aggregate_fitness_path: Optional[Path] = None
    trimmed_fitness_path: Optional[Path] = None
    sequence_processes: Optional[int] = None
    lineage_column: str = "clade"

    def resolve_output_dir(self, season: SeasonConfig) -> Path:
        hemi_token = season.hemisphere.lower()
        return self.output_root / f"{self.subtype}_{hemi_token}" / f"season_{season.year}"


def _build_date_suffix(year: int, hemisphere: str) -> str:
    if hemisphere.lower() == "north":
        return f"{year}0131"
    return f"{year-1}0831"


def _load_evescape_tables(evescape_dir: Path, prefix: str, date_suffix: str) -> Dict[str, pd.DataFrame]:
    mutations_path = evescape_dir / f"{prefix}_evescape_{date_suffix}.csv"
    sites_path = evescape_dir / f"{prefix}_evescape_sites_{date_suffix}.csv"
    if not mutations_path.exists():
        raise FileNotFoundError(f"EVEscape突变文件缺失: {mutations_path}")
    if not sites_path.exists():
        raise FileNotFoundError(f"EVEscape位点文件缺失: {sites_path}")
    return {
        "mutations": pd.read_csv(mutations_path),
        "sites": pd.read_csv(sites_path),
    }


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# Prepare the fitness table by keeping the top-ranked row for each clade.
# 针对每个谱系保留首条记录并按初始顺序输出。
def _prepare_clade_ranked_fitness(fitness_df: pd.DataFrame) -> pd.DataFrame:
    target_columns = [
        "subtype",
        "hemisphere",
        "year",
        "fitness",
        "clade",
    ]
    if fitness_df.empty:
        return pd.DataFrame(columns=target_columns)

    prepared = fitness_df.copy()
    clade_series = prepared.get("clade", pd.Series([""] * len(prepared)))
    prepared["clade"] = clade_series.fillna("").astype(str).str.strip()
    valid_mask = ~prepared["clade"].str.lower().isin({"", "unassigned", "unknown"})
    prepared = prepared[valid_mask]

    if prepared.empty:
        return pd.DataFrame(columns=target_columns)

    if "rank" in prepared.columns:
        prepared = prepared.sort_values(
            by=["rank", "fitness" if "fitness" in prepared.columns else "clade"],
            ascending=[True, False if "fitness" in prepared.columns else True],
            kind="mergesort",
        )
    elif "fitness" in prepared.columns:
        prepared = prepared.sort_values(by="fitness", ascending=False, kind="mergesort")

    prepared = prepared.drop_duplicates(subset=["clade"], keep="first").reset_index(drop=True)

    for column in target_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    return prepared[target_columns]


# Run Step1 + Step2 + Step3 for a single subtype.
# 针对单个亚型依次执行三个处理阶段。
# Returns a dict containing the aggregated dataframe (key `data`) and any
# generated file paths under descriptive keys.
# 返回的字典包含汇总数据帧（键为 `data`）以及输出文件路径。
def run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    generated: Dict[str, object] = {}

    # Prepare the subtype-wide sequence table once.
    # 仅构建一次全亚型序列表以复用。
    sequence_table = generate_sequence_table(
        subtype=config.subtype,
        fasta_path=config.fasta_path,
        info_path=config.info_path,
        cutoff_date=config.sequence_cutoff,
        processes=config.sequence_processes,
    ).copy()
    sequence_table["collection_date"] = pd.to_datetime(sequence_table["collection_date"])
    sequence_table["submission_date"] = pd.to_datetime(sequence_table["submission_date"])

    epi_cache: Dict[Path, pd.DataFrame] = {}
    aggregated_frames: List[pd.DataFrame] = []
    prevalence_tables: List[pd.DataFrame] = []

    for season in config.seasons:
        epi_df = epi_cache.get(season.epi_csv)
        if epi_df is None:
            epi_df = pd.read_csv(season.epi_csv)
            epi_cache[season.epi_csv] = epi_df

        (
            mutation_pred_df,
            mutation_analysis_df,
            prevalence_df,
            theta_info,
        ) = run_prediction_pipeline(
            seq_df=sequence_table,
            epi_df=epi_df,
            predict_season=season.year,
            semisphere=season.hemisphere,
            subtype=config.subtype,
            theta_range=config.theta_range,
            log_dir=None,
        )

        # Normalise theta info to builtin floats for easier serialization/debugging.
        # 将 theta 信息转换为内置浮点类型，便于序列化与调试。
        theta_info = {
            "predict_season": int(theta_info.get("predict_season", season.year)),
            "best_theta": float(theta_info.get("best_theta", 0.0)),
            "best_r2": float(theta_info.get("best_r2", 0.0)),
            "years_used": int(theta_info.get("years_used", 0)),
        }

        date_suffix = _build_date_suffix(season.year, season.hemisphere)
        evescape_tables = _load_evescape_tables(
            config.evescape_dir, config.evescape_prefix, date_suffix
        )

        fitness_df = compute_fitness_for_prediction(
            year=season.year,
            virus_type=config.subtype,
            hemisphere=season.hemisphere,
            sequence_df=sequence_table,
            risk_mutations_df=mutation_analysis_df,
            mutation_prediction_df=mutation_pred_df,
            mutations_df=evescape_tables["mutations"],
            sites_df=evescape_tables["sites"],
        )

        if not fitness_df.empty:
            trimmed_frame = fitness_df[
                [
                    "subtype",
                    "hemisphere",
                    "year",
                    "rank",
                    "risk_mutation_group",
                    "mutation_count",
                    "fitness",
                    "clade",
                ]
            ].copy()
            aggregated_frames.append(trimmed_frame)

        prevalence_with_meta = prevalence_df.copy()
        prevalence_with_meta.insert(0, "predict_season", season.year)
        prevalence_with_meta.insert(0, "hemisphere", season.hemisphere)
        prevalence_with_meta.insert(0, "subtype", config.subtype)
        prevalence_tables.append(prevalence_with_meta)

        out_dir = config.resolve_output_dir(season)
        out_dir.mkdir(parents=True, exist_ok=True)

        fitting_df = pd.DataFrame([theta_info])
        theta_output = fitting_df.drop(columns=["best_r2"], errors="ignore")
        _write_dataframe(theta_output, out_dir / "theta_fitting.csv")
        _write_dataframe(mutation_pred_df, out_dir / "mutation_prediction.csv")
        prevalence_output = prevalence_df.drop(columns=["dominant_mutation"], errors="ignore")
        _write_dataframe(prevalence_output, out_dir / "prevalence_result.csv")
        clade_fitness = _prepare_clade_ranked_fitness(fitness_df)
        _write_dataframe(clade_fitness, out_dir / "fitness.csv")

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
    if aggregated_frames:
        combined = (
            pd.concat(aggregated_frames, ignore_index=True)
            .sort_values(["subtype", "hemisphere", "year", "rank"])
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=trimmed_columns)
    combined["hemisphere"] = combined["hemisphere"].astype(str).str.lower()
    generated["data"] = combined

    generated["sequence_table"] = sequence_table.copy()
    generated["config_meta"] = {
        "subtype": config.subtype,
        "fasta_name": config.fasta_path.name,
    }

    generated["prevalence_result"] = (
        pd.concat(prevalence_tables, ignore_index=True) if prevalence_tables else pd.DataFrame()
    )

    if config.aggregate_fitness_path is not None:
        _write_dataframe(combined, config.aggregate_fitness_path)
        generated["aggregate_fitness"] = config.aggregate_fitness_path

    if config.trimmed_fitness_path is not None:
        _write_dataframe(combined, config.trimmed_fitness_path)
        generated["trimmed_fitness"] = config.trimmed_fitness_path

    return generated


__all__ = ["PipelineConfig", "SeasonConfig", "run_pipeline"]
