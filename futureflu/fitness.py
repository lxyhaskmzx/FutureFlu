from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# Count the number of mutations inside a grouped label.
# 统计风险突变组合中包含的突变数量。
def count_mutations(mutation_group: str) -> int:
    if pd.isna(mutation_group) or mutation_group == "":
        return 0
    return len(mutation_group.split(","))


# Compute the mutual-information proxy for a single mutation.
# 计算单个突变的互信息指标。
def calculate_single_mutation_mi(mutation: str, df: pd.DataFrame) -> float:
    mutation_groups = df[df["risk_mutation_group"].str.contains(mutation, na=False)]
    total_occurrences = mutation_groups["mutation_count"].sum()
    if total_occurrences == 0:
        return 0
    solo_occurrences = mutation_groups[mutation_groups["mutation_count"] == 1][
        "mutation_count"
    ].sum()
    return solo_occurrences / total_occurrences


# Compute the normalised mutual information for a mutation group.
# 计算突变组的标准化互信息。
def calculate_group_mutual_information(mutation_matrix: pd.DataFrame) -> float:
    if mutation_matrix.shape[1] == 0 or mutation_matrix.shape[1] == 1:
        return 0

    n = len(mutation_matrix)
    marginal_probs = mutation_matrix.mean()
    patterns = mutation_matrix.apply(lambda x: "".join(x.astype(str)), axis=1)
    joint_counts = patterns.value_counts()

    mi_value = 0.0
    for pattern, count in joint_counts.items():
        p_joint = count / n
        if p_joint > 0:
            binary_pattern = [int(b) for b in pattern]
            p_indep = 1
            for i, mut in enumerate(mutation_matrix.columns):
                p_i = (
                    marginal_probs[mut]
                    if binary_pattern[i] == 1
                    else (1 - marginal_probs[mut])
                )
                p_indep *= p_i
            if p_indep > 0:
                mi_value += p_joint * np.log2(p_joint / p_indep)

    normalized_mi = mi_value / mutation_matrix.shape[1]
    return float(normalized_mi)


# Build a mutation matrix without filtering the source sequences.
# 生成不筛选序列的突变矩阵。
def get_mutation_matrix_simple(seq_df: pd.DataFrame, mutations: List[str]) -> pd.DataFrame:
    mutation_matrix = pd.DataFrame()
    for mut in mutations:
        site = int("".join(filter(str.isdigit, mut)))
        target_aa = mut[-1]
        col_name = f"X{site}"
        if col_name in seq_df.columns:
            mutation_matrix[mut] = (seq_df[col_name] == target_aa).astype(int)
        else:
            mutation_matrix[mut] = 0
    return mutation_matrix


# Build a mutation matrix filtered by true supersets for lineage mapping.
# 基于真超集筛选生成用于谱系映射的突变矩阵。
def get_mutation_matrix(
    seq_df: pd.DataFrame,
    mutations: List[str],
    all_mutation_groups: Sequence[str],
) -> pd.DataFrame:
    mutation_matrix = pd.DataFrame()
    matching_sequences = seq_df.copy()

    for mut in mutations:
        site = int("".join(filter(str.isdigit, mut)))
        target_aa = mut[-1]
        col_name = f"X{site}"
        if col_name in seq_df.columns:
            matching_sequences = matching_sequences.loc[matching_sequences[col_name] == target_aa]

    for other_group in all_mutation_groups:
        other_mutations = other_group.split(",")
        if set(other_mutations) > set(mutations):
            extra_mutations = set(other_mutations) - set(mutations)
            for mut in extra_mutations:
                site = int("".join(filter(str.isdigit, mut)))
                target_aa = mut[-1]
                col_name = f"X{site}"
                if col_name in seq_df.columns:
                    matching_sequences = matching_sequences.loc[matching_sequences[col_name] != target_aa]

    for mut in mutations:
        site = int("".join(filter(str.isdigit, mut)))
        target_aa = mut[-1]
        col_name = f"X{site}"
        if col_name in seq_df.columns:
            mutation_matrix[mut] = (matching_sequences[col_name] == target_aa).astype(int)
    return mutation_matrix


# Calculate the escape value with optional Victoria-specific adjustments.
# 计算逃逸值并在需要时应用 Victoria 特有调整。
def calculate_total_escape_value(
    mutation_group: str,
    mutation_escape: Dict[str, float],
    virus_type: str,
    sites_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
) -> float:
    if pd.isna(mutation_group):
        return 0.0

    mutations = mutation_group.split(",")
    total_escape = 0.0
    need_adjustment = (
        virus_type == "Victoria"
        and len(sites_df) != 585
        and len(mutations_df) != 585
    )

    for mut in mutations:
        mut = mut.strip()
        site = int("".join(filter(str.isdigit, mut)))

        if need_adjustment and site >= 177:
            adjusted_site = site - (585 - len(sites_df))
            adjusted_mut = f"{adjusted_site}{mut[-1]}"

            if adjusted_mut in mutation_escape:
                total_escape += mutation_escape[adjusted_mut]
            elif str(adjusted_site) in mutation_escape:
                total_escape += mutation_escape[str(adjusted_site)]
        else:
            if mut in mutation_escape:
                total_escape += mutation_escape[mut]
            elif str(site) in mutation_escape:
                total_escape += mutation_escape[str(site)]

    return total_escape


# Calculate the average predicted prevalence for a mutation group.
# 计算突变组合的平均预测流行度。
def calculate_prevalence(
    mutation_group: str,
    mutation_prevalence: Dict[str, float],
) -> float:
    if pd.isna(mutation_group):
        return 0.0

    mutations = mutation_group.split(",")
    prevalence_sum = 0.0
    for mut in mutations:
        mut = mut.strip()
        if mut in mutation_prevalence:
            prevalence_sum += mutation_prevalence[mut]
    avg_prevalence = prevalence_sum / len(mutations) if mutations else 0.0
    return avg_prevalence


# Infer a clade label by testing true supersets of the mutation group.
# 通过真超集判定推断突变组所属的谱系标签。
def get_clade_info(
    mutation_group: str,
    sequence_df: pd.DataFrame,
    all_mutation_groups: Sequence[str],
) -> str:
    if pd.isna(mutation_group):
        return "Unknown"

    mutations = mutation_group.split(",")
    matching_sequences = sequence_df.copy()

    for mut in mutations:
        mut = mut.strip()
        site = int("".join(filter(str.isdigit, mut)))
        target_aa = mut[-1]
        col_name = f"X{site}"

        if col_name in sequence_df.columns:
            matching_sequences = matching_sequences.loc[matching_sequences[col_name] == target_aa]

    for other_group in all_mutation_groups:
        other_mutations = other_group.split(",")
        if set(other_mutations) > set(mutations):
            extra_mutations = set(other_mutations) - set(mutations)
            for mut in extra_mutations:
                mut = mut.strip()
                site = int("".join(filter(str.isdigit, mut)))
                target_aa = mut[-1]
                col_name = f"X{site}"
                if col_name in sequence_df.columns:
                    matching_sequences = matching_sequences.loc[matching_sequences[col_name] != target_aa]

    if len(matching_sequences) == 0:
        return "Unknown"

    clade_series = matching_sequences["clade"].fillna("Unknown").astype(str)
    clade_counts = clade_series.value_counts()
    if len(clade_counts) == 0:
        return "Unknown"

    placeholder_values = {"unknown", "unassigned", ""}
    for clade_value in clade_counts.index:
        if clade_value.strip().lower() not in placeholder_values:
            return clade_value
    return clade_counts.index[0]


# Calculate fitness via z-score, sigmoid-log, and temperature weighting.
# 通过 z 分数、sigmoid-对数和温度加权流程计算适应度。
def calculate_fitness(results_list: List[dict], virus_type: str) -> pd.DataFrame:
    df = pd.DataFrame(results_list).fillna(0)

    def z_score_normalize(x):
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x)
        return (x - mean) / std

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    temperatures = {
        "H3N2": {"total_escape": 10, "predicted_prevalence": 9, "mutual_information": 0.3},
        "H1N1": {"total_escape": 5.0, "predicted_prevalence": 1.8, "mutual_information": 1.1},
        "Victoria": {"total_escape": 1.3, "predicted_prevalence": 1.1, "mutual_information": 0.6},
    }

    current_temperatures = temperatures[virus_type]

    for feature in ["total_escape", "predicted_prevalence", "mutual_information"]:
        normalized = z_score_normalize(df[feature])
        sigmoid_transformed = sigmoid(normalized / current_temperatures[feature])
        df[f"{feature}_processed"] = np.log(sigmoid_transformed)

    df["fitness"] = (
        df["total_escape_processed"]
        + df["predicted_prevalence_processed"]
        + df["mutual_information_processed"]
    )
    return df


# Compute fitness for a single configuration.
# 执行单个组合的适应度计算。
def compute_fitness_for_prediction(
    *,
    year: int,
    virus_type: str,
    hemisphere: str,
    sequence_df: pd.DataFrame,
    risk_mutations_df: pd.DataFrame,
    mutation_prediction_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    if risk_mutations_df.empty or "risk_mutation_group" not in risk_mutations_df.columns:
        return pd.DataFrame(
            columns=[
                "risk_mutation_group",
                "mutation_count",
                "total_escape",
                "predicted_prevalence",
                "mutual_information",
                "total_escape_processed",
                "predicted_prevalence_processed",
                "mutual_information_processed",
                "fitness",
                "clade",
                "subtype",
                "hemisphere",
                "year",
                "rank",
            ]
        )

    df = risk_mutations_df.copy()
    df["mutation_count"] = df["risk_mutation_group"].apply(count_mutations)
    df_filtered = df[df["mutation_count"] > 0]

    if df_filtered.empty:
        raise ValueError("No mutation groups remain after initial filtering.")

    q1 = df_filtered["mutation_count"].quantile(0.25)
    q3 = df_filtered["mutation_count"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    lower_bound = q1 - 3 * iqr

    non_outliers = df_filtered[
        (df_filtered["mutation_count"] <= upper_bound)
        & (df_filtered["mutation_count"] >= lower_bound)
    ]
    if non_outliers.empty:
        raise ValueError("No mutation groups remain after outlier filtering.")

    all_mutation_groups = non_outliers["risk_mutation_group"].dropna().tolist()

    sequence_df = sequence_df.copy()
    sequence_df["collection_date"] = pd.to_datetime(sequence_df["collection_date"])
    sequence_df["submission_date"] = pd.to_datetime(sequence_df["submission_date"])

    if hemisphere.lower() == "north":
        start_date = f"{year-1}-09-01"
        end_date = f"{year}-02-01"
    else:
        start_date = f"{year-1}-02-01"
        end_date = f"{year-1}-09-01"

    if virus_type == "H1N1":
        filtered_seq_df_mi = sequence_df[
            (sequence_df["collection_date"] < end_date)
            & (sequence_df["submission_date"] < end_date)
        ]
    else:
        filtered_seq_df_mi = sequence_df[
            (sequence_df["collection_date"] >= start_date)
            & (sequence_df["collection_date"] < end_date)
            & (sequence_df["submission_date"] < end_date)
        ]

    filtered_seq_df_lineage = sequence_df[
        (sequence_df["collection_date"] >= start_date)
        & (sequence_df["collection_date"] < end_date)
        & (sequence_df["submission_date"] < end_date)
    ]

    mutation_escape: Dict[str, float] = {}
    mutations_min_escape = mutations_df["evescape"].min()
    sites_min_escape = sites_df["evescape"].min()

    for _, row in mutations_df.iterrows():
        mutation = f"{row['i']}{row['mut']}"
        mutation_escape[mutation] = row["evescape"] - mutations_min_escape

    for _, row in sites_df.iterrows():
        site = str(row["i"])
        mutation_escape[site] = row["evescape"] - sites_min_escape

    mutation_prevalence = dict(
        zip(mutation_prediction_df["risk_mutation"], mutation_prediction_df["predicted_prevalence"])
    )

    results: List[dict] = []
    for _, row in non_outliers.iterrows():
        mutation_group = row["risk_mutation_group"]
        total_escape_value = calculate_total_escape_value(
            mutation_group=mutation_group,
            mutation_escape=mutation_escape,
            virus_type=virus_type,
            sites_df=sites_df,
            mutations_df=mutations_df,
        )
        predicted_prevalence = calculate_prevalence(
            mutation_group=mutation_group,
            mutation_prevalence=mutation_prevalence,
        )
        count = row["mutation_count"]

        if pd.isna(mutation_group):
            mutual_info = 0
        else:
            mutations = [m.strip() for m in mutation_group.split(",")]
            if len(mutations) == 1:
                mutual_info = calculate_single_mutation_mi(mutations[0], non_outliers)
            else:
                mut_matrix = get_mutation_matrix_simple(filtered_seq_df_mi, mutations)
                mutual_info = calculate_group_mutual_information(mut_matrix)

        lineage_label = get_clade_info(
            mutation_group, filtered_seq_df_lineage, all_mutation_groups
        )

        results.append(
            {
                "risk_mutation_group": mutation_group,
                "mutation_count": count,
                "total_escape": total_escape_value,
                "predicted_prevalence": predicted_prevalence,
                "mutual_information": mutual_info,
                "clade": lineage_label,
                "subtype": virus_type,
                "hemisphere": hemisphere,
                "year": year,
            }
        )

    results_df = calculate_fitness(results, virus_type)
    results_df = results_df.sort_values(by="fitness", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)
    return results_df


__all__ = ["compute_fitness_for_prediction"]
