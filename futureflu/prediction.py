from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# Compute amino-acid site prevalence for a subtype and season.
# 计算特定亚型与季度的氨基酸位点流行度。
def site_prevalence(seq: pd.DataFrame, predict_season: int, semisphere: str, subtype: str) -> pd.DataFrame:
    ha1_ranges = {
        "H3N2": (17, 345),
        "H1N1": (18, 344),
        "Victoria": (15, 362),
    }
    ha1_range = ha1_ranges[subtype]

    years = sorted([y for y in seq["season"].unique() if y > 2009 and y < predict_season])

    amino_acids = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "-",
    ]

    seq = seq.copy()
    seq["submission_date"] = seq["submission_date"].fillna(seq["collection_date"])
    seq["collection_date"] = pd.to_datetime(seq["collection_date"])
    seq["submission_date"] = pd.to_datetime(seq["submission_date"])

    site_columns = seq.filter(regex="^X\\d+").columns
    site_columns = [
        col
        for col in site_columns
        if ha1_range[0] <= int("".join(filter(str.isdigit, col))) <= ha1_range[1]
    ]

    for col in site_columns:
        seq[col] = seq[col].str.upper()

    annual_data = {}
    prevalence_data: List[pd.Series] = []

    def get_consensus_sequence(data: pd.DataFrame, site_columns: List[str]) -> dict:
        consensus = {}
        for col in site_columns:
            valid_data = data[col][data[col] != "X"]
            if len(valid_data) > 0:
                consensus[col] = valid_data.mode().iloc[0]
        return consensus

    def calculate_significance(new_aa_current, new_aa_prev, total_current, total_prev):
        from scipy import stats
        import numpy as np

        observed = np.array(
            [
                [new_aa_current, total_current - new_aa_current],
                [new_aa_prev, total_prev - new_aa_prev],
            ]
        )

        if np.any(observed == 0):
            return 0.0

        _, p_value = stats.chi2_contingency(observed)[:2]
        return p_value

    for year in years:
        if semisphere == "North":
            start = f"{year}-09-01"
            end = f"{year+1}-02-01"
            submission_end = f"{predict_season}-02-01"
        else:
            start = f"{year}-02-01"
            end = f"{year}-09-01"
            submission_end = f"{predict_season-1}-09-01"

        season_data = seq[
            (seq["collection_date"] >= start)
            & (seq["collection_date"] < end)
            & (seq["submission_date"] < submission_end)
        ]

        year_freq_data = {}
        year_counts_data = {}
        for site_col in site_columns:
            valid_data = season_data[site_col][season_data[site_col] != "X"]
            freq = valid_data.value_counts(normalize=True)
            counts = valid_data.value_counts()
            for aa in amino_acids:
                col_name = f"{site_col}{aa}"
                year_freq_data[col_name] = freq.get(aa, 0.0)
            year_counts_data[site_col] = counts

        prevalence_data.append(pd.Series(year_freq_data, name=year))

        annual_data[year] = {
            "freqs": year_freq_data,
            "counts": year_counts_data,
            "dominant_mutations": [],
        }

        if year != years[0]:
            if semisphere == "North":
                prev_start = f"{year-1}-09-01"
                prev_end = f"{year}-02-01"
            else:
                prev_start = f"{year-1}-02-01"
                prev_end = f"{year-1}-09-01"

            prev_data = seq[
                (seq["collection_date"] >= prev_start)
                & (seq["collection_date"] < prev_end)
                & (seq["submission_date"] < submission_end)
            ]

            current_consensus = get_consensus_sequence(season_data, site_columns)
            prev_consensus = get_consensus_sequence(prev_data, site_columns)

            prev_counts_data = {}
            for site_col in site_columns:
                valid_data = prev_data[site_col][prev_data[site_col] != "X"]
                prev_counts_data[site_col] = valid_data.value_counts()

            identified_mutations = []
            for site_col in site_columns:
                if (
                    site_col in current_consensus
                    and site_col in prev_consensus
                    and current_consensus[site_col] != prev_consensus[site_col]
                ):
                    old_aa = prev_consensus[site_col]
                    new_aa = current_consensus[site_col]
                    current_counts = year_counts_data[site_col]
                    prev_counts = prev_counts_data[site_col]

                    new_aa_current = current_counts.get(new_aa, 0)
                    new_aa_prev = prev_counts.get(new_aa, 0)
                    total_current = sum(current_counts)
                    total_prev = sum(prev_counts)

                    p_value = calculate_significance(
                        new_aa_current, new_aa_prev, total_current, total_prev
                    )
                    if p_value < 0.05:
                        mut = f"{site_col[1:]}{new_aa}"
                        identified_mutations.append(mut)

            annual_data[year]["dominant_mutations"] = sorted(identified_mutations)

    prevalence_result = pd.concat(prevalence_data, axis=1).T
    prevalence_result["dominant_mutation"] = [
        ", ".join(annual_data[y]["dominant_mutations"]) for y in years
    ]
    mutation_columns = [
        col
        for col in prevalence_result.columns
        if col not in ["dominant_mutation", "dominant_clades"]
    ]
    column_order = mutation_columns + ["dominant_mutation"]
    prevalence_result = prevalence_result[column_order]
    return prevalence_result.reset_index().rename(columns={"index": "season"})


# Compute annual gsum totals for every theta value.
# 计算每个 theta 参数对应的年度 gsum。
def gmeasure(prev_data: pd.DataFrame, theta_range: Sequence[float]) -> pd.DataFrame:
    valid_years = prev_data["season"].tolist()
    year_theta_gsum = {year: {theta: 0.0 for theta in theta_range} for year in valid_years}

    for theta in theta_range:
        for col in prev_data.columns:
            if col.startswith("X"):
                values = prev_data[col].fillna(0).values
                n_years = len(values)

                mut = np.zeros(n_years, dtype=int)
                start = 0

                for r in range(n_years):
                    if values[r] >= theta and np.any(values[start:r] < theta):
                        low_pos = np.where(values[:r] < theta)[0]
                        if low_pos.size > 0:
                            a = low_pos[-1]
                            mut[a + 1 : r + 1] = 1
                            start = r + 1

                yearly_gsum = values * mut
                for idx, year in enumerate(valid_years):
                    year_theta_gsum[year][theta] += yearly_gsum[idx]

    gsum_df = pd.DataFrame.from_dict(year_theta_gsum, orient="index")
    gsum_df.columns = [f"theta={theta:.2f}" for theta in theta_range]
    gsum_df = gsum_df.reset_index().rename(columns={"index": "season"})
    return gsum_df


# Provide a placeholder logger for theta fitting when logging is requested.
# 当需要日志时提供一个占位的 theta 拟合记录器。
def setup_logging(base_path) -> Callable:
    Path(base_path).mkdir(parents=True, exist_ok=True)
    return _noop_logger


# Fit regression models to identify the optimal theta grid.
# 拟合回归以找到最优的 theta 搜索网格。
def fit_regression(
    input_gmeasure: pd.DataFrame,
    input_epi_data: pd.DataFrame,
    log_print: Callable[[object], None],
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    gmeasure_df = input_gmeasure.set_index("season").copy()
    epi_data = input_epi_data.copy()

    epi_data["Season"] = epi_data["Season"].astype(int)
    gmeasure_df.index = gmeasure_df.index.astype(str)

    valid_gmeasure_years = [y for y in gmeasure_df.index if y.isdigit()]
    gmeasure_years = pd.to_numeric(valid_gmeasure_years, errors="coerce")
    gmeasure_years = gmeasure_years[~np.isnan(gmeasure_years)].astype(int).tolist()

    common_years = sorted(set(epi_data["Season"]).intersection(gmeasure_years))
    if not common_years and not epi_data.empty:
        min_epi_year = epi_data["Season"].min()
        filler_template = epi_data.iloc[0].to_dict()
        filler_rows = []
        for year in sorted(y for y in gmeasure_years if y < min_epi_year):
            row = filler_template.copy()
            row["Season"] = year
            filler_rows.append(row)
        if filler_rows:
            epi_data = pd.concat([pd.DataFrame(filler_rows), epi_data], ignore_index=True)
            common_years = sorted(set(epi_data["Season"]).intersection(gmeasure_years))

    best_overall = {
        "theta": None,
        "r_squared": -float("inf"),
        "years_used": None,
        "start_year": None,
    }

    max_year = max(common_years)
    log_print("\nScanning seasonal windows for regression fits:")

    for years_back in range(3, len(common_years) + 1):
        start_year = max_year - years_back + 1
        current_years = [y for y in common_years if start_year <= y <= max_year]

        if len(current_years) < 3:
            continue

        log_print(f"\nTrying {years_back} seasonal years ({start_year}-{max_year}):")

        current_epi = epi_data[epi_data["Season"].isin(current_years)]
        current_seq = gmeasure_df.loc[[str(y) for y in current_years]]

        results = []
        for col in current_seq.columns:
            if not col.startswith("theta="):
                continue

            theta = float(col.split("=")[1])

            X = current_seq[col].values.reshape(-1, 1)
            y = current_epi["Positivity_Rate"].values

            X = np.hstack([np.ones((X.shape[0], 1)), X])

            try:
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                y_pred = X @ beta

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                results.append({"Theta": theta, "R_squared": round(r_squared, 3)})
            except np.linalg.LinAlgError:
                continue

        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(["R_squared", "Theta"], ascending=[False, True])
            best_current = results_df.iloc[0]

            log_print(
                f"Current best - Theta: {best_current['Theta']:.2f}, R²: {best_current['R_squared']:.3f}"
            )

            if best_current["R_squared"] > best_overall["r_squared"]:
                best_overall["theta"] = best_current["Theta"]
                best_overall["r_squared"] = best_current["R_squared"]
                best_overall["years_used"] = years_back
                best_overall["start_year"] = start_year

    if best_overall["theta"] is not None:
        log_print("\n=== Final regression summary ===")
        log_print(
            f"Year span: {best_overall['start_year']}-{max(common_years)} "
            f"(total {best_overall['years_used']} years)"
        )
        log_print(f"Best theta: {best_overall['theta']:.2f}")
        log_print(f"Best R²: {best_overall['r_squared']:.3f}")
        return (
            best_overall["theta"],
            best_overall["r_squared"],
            best_overall["years_used"],
        )
    else:
        log_print("\nWarning: no valid regression window produced a fit")
        return None, None, None


# Predict high-risk mutations for a given season.
# 预测指定年份的高风险突变。
def predict_mutations(
    predict_season: int,
    theta: float,
    prev_data: pd.DataFrame,
    mutation_history: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        columns=["predict_season", "risk_mutation", "previous_prevalence", "predicted_prevalence"]
    )
    risk_muts = []

    mutation_columns = [col for col in prev_data.columns if col.startswith("X")]
    historical_data = prev_data[prev_data.index < predict_season][mutation_columns]

    mutation_dominant_years = {}
    if "dominant_mutation" in mutation_history.columns:
        for _, row in mutation_history.iterrows():
            year = row["season"]
            if year >= predict_season:
                continue

            dominant_muts = row["dominant_mutation"]
            if isinstance(dominant_muts, str) and dominant_muts.strip():
                for mut in dominant_muts.split(","):
                    mut = mut.strip()
                    if mut:
                        if mut not in mutation_dominant_years or year > mutation_dominant_years[mut]:
                            mutation_dominant_years[mut] = year

    pred_prev = np.zeros(len(historical_data.columns))

    for col_idx, col in enumerate(historical_data.columns):
        freqs = historical_data[col].values[-2:]

        try:
            if len(freqs) >= 2:
                delta = freqs[-1] - freqs[-2]
            else:
                delta = 0

            predicted = freqs[-1] + delta if len(freqs) > 0 else 0
            pred_prev[col_idx] = np.clip(predicted, 0, 1)

        except Exception:
            pred_prev[col_idx] = 0

    for col_idx, col in enumerate(historical_data.columns):
        pred = pred_prev[col_idx]
        formatted_mut = col[1:] if col.startswith("X") else col

        if formatted_mut in mutation_dominant_years:
            start_year = mutation_dominant_years[formatted_mut]
            historical_years = prev_data[prev_data.index < predict_season].index.tolist()
            try:
                start_idx = historical_years.index(start_year)
                freqs = historical_data[col].values[start_idx:]
            except ValueError:
                freqs = historical_data[col].values
        else:
            freqs = historical_data[col].values

        condition1 = (pred >= theta) and np.any(freqs < theta)

        condition2 = False
        if theta / 10 >= 0.01:
            condition2 = (pred >= theta / 10) and np.any(freqs < theta / 10)
        elif theta * 10 < 1:
            condition2 = (pred >= theta * 10) and np.any(freqs < theta * 10)

        if condition1 or condition2:
            risk_muts.append(col)

    year_records = []
    for mut in risk_muts:
        formatted_mut = mut[1:] if mut.startswith("X") else mut
        prev_prev = historical_data[mut].iloc[-1] if len(historical_data[mut]) > 0 else 0
        mut_idx = list(historical_data.columns).index(mut)
        predicted_prev = pred_prev[mut_idx]

        year_records.append(
            {
                "predict_season": predict_season,
                "risk_mutation": formatted_mut,
                "previous_prevalence": round(prev_prev, 4),
                "predicted_prevalence": round(predicted_prev, 4),
            }
        )

    if year_records:
        predictions = pd.concat([predictions, pd.DataFrame(year_records)], ignore_index=True)

    return predictions


# Analyse how predicted risk mutations distribute across sequences.
# 分析预测风险突变在序列中的分布情况。
def analyze_risk_mutations(
    sequences_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    predict_season: int,
    semisphere: str,
) -> pd.DataFrame:
    if semisphere == "North":
        start = f"{predict_season-1}-09-01"
        end = f"{predict_season}-02-01"
        submission_end = f"{predict_season}-02-01"
    else:
        start = f"{predict_season-1}-02-01"
        end = f"{predict_season-1}-09-01"
        submission_end = f"{predict_season-1}-09-01"

    sequences_df = sequences_df.copy()
    sequences_df["collection_date"] = pd.to_datetime(sequences_df["collection_date"])
    sequences_df["submission_date"] = pd.to_datetime(sequences_df["submission_date"])

    date_mask = (
        (sequences_df["collection_date"] >= start)
        & (sequences_df["collection_date"] < end)
        & (sequences_df["submission_date"] < submission_end)
    )
    filtered_sequences = sequences_df[date_mask]

    def parse_mutation(mutation_str):
        position = int("".join(filter(str.isdigit, mutation_str)))
        amino_acid = mutation_str.replace(str(position), "")
        return position, amino_acid

    risk_mutations = [parse_mutation(mut) for mut in mutations_df["risk_mutation"]]

    def find_risk_mutations(row):
        mutations_found = []
        for pos, aa in risk_mutations:
            column_name = f"X{pos}"
            if column_name in row:
                if aa == "-":
                    if pd.isna(row[column_name]) or row[column_name] == "-":
                        mutations_found.append(f"{pos}{aa}")
                else:
                    if row[column_name] == aa:
                        mutations_found.append(f"{pos}{aa}")
        return ",".join(sorted(mutations_found)) if mutations_found else None

    filtered_sequences = filtered_sequences.copy()
    filtered_sequences["risk_mutation_group"] = filtered_sequences.apply(
        find_risk_mutations, axis=1
    )

    filtered_sequences = filtered_sequences[filtered_sequences["risk_mutation_group"].notna()]

    mutation_counts = (
        filtered_sequences["risk_mutation_group"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "risk_mutation_group"})
    )

    if mutation_counts.empty:
        return pd.DataFrame(columns=["risk_mutation_group", "count"])
    return mutation_counts


def _noop_logger(*args, **kwargs):

    return None


# Run the integrated prediction workflow and return prevalence results.
# 执行位点流行度、theta 拟合与风险突变预测的整合流程，并返回流行度结果。
def run_prediction_pipeline(
    seq_df: pd.DataFrame,
    epi_df: pd.DataFrame,
    predict_season: int,
    semisphere: str,
    subtype: str,
    theta_range: Sequence[float],
    log_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prev_result = site_prevalence(seq_df, predict_season, semisphere, subtype)
    gmeasure_result = gmeasure(prev_result, theta_range)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_print = setup_logging(log_dir)
    else:
        # Disable file logging when no output directory is provided.
        # 未提供输出目录时禁用文件日志写入。
        log_print = _noop_logger

    best_theta, best_r2, years_used = fit_regression(gmeasure_result, epi_df, log_print)
    if best_theta is None:
        best_theta = float(theta_range[0])
        best_r2 = 0.0
        years_used = 0

    mutation_pred = predict_mutations(
        predict_season=predict_season,
        theta=best_theta,
        prev_data=prev_result.set_index("season"),
        mutation_history=prev_result[["season", "dominant_mutation"]],
    )

    mutation_analysis = analyze_risk_mutations(
        sequences_df=seq_df,
        mutations_df=mutation_pred,
        predict_season=predict_season,
        semisphere=semisphere,
    )

    theta_info = {
        "predict_season": predict_season,
        "best_theta": best_theta,
        "best_r2": best_r2,
        "years_used": years_used,
    }
    return mutation_pred, mutation_analysis, prev_result, theta_info


__all__ = [
    "site_prevalence",
    "gmeasure",
    "setup_logging",
    "fit_regression",
    "predict_mutations",
    "analyze_risk_mutations",
    "run_prediction_pipeline",
]
