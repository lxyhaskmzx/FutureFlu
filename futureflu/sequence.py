from __future__ import annotations

import itertools
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


import warnings

warnings.filterwarnings("ignore")


# Helper routine for parallel sequence processing.
# 并行处理序列数据的辅助函数。
def parallel_sequence_processing(
    chunk_data: Tuple[List[str], List[str], pd.DataFrame],
    cutoff_date: str,
    subtype: str = "H3N2",
) -> List[dict]:
    sequences, names, info_data = chunk_data
    valid_rows = []
    for seq, name in zip(sequences, names):
        parts = name.split("|")
        isolate_id = parts[0]
        victoria_segment_id = parts[1] if len(parts) > 1 else parts[0]

        if seq.count("-") > 3:
            continue

        if subtype == "Victoria":
            iso_id_col = (
                info_data["Isolate_Id"].fillna("")
                if "Isolate_Id" in info_data.columns
                else pd.Series([""] * len(info_data))
            )
            mask = iso_id_col.astype(str) == victoria_segment_id
        else:
            mask = info_data["Isolate_Id"] == isolate_id

        collection_info = info_data.loc[mask, "Collection_Date"].values
        submission_info = info_data.loc[mask, "Submission_Date"].values
        clade_info = info_data.loc[mask, "Clade"].values
        isolate_name_info = (
            info_data.loc[mask, "Isolate_Name"].values
            if "Isolate_Name" in info_data.columns
            else np.array([], dtype=object)
        )

        if len(collection_info) > 0 and pd.notna(collection_info[0]):
            collection_str = str(collection_info[0])
            submission_str = (
                str(submission_info[0])
                if len(submission_info) > 0 and pd.notna(submission_info[0])
                else ""
            )

            if submission_str:
                parts = submission_str.split("-")
                if len(parts) == 1:
                    submission_str = f"{submission_str}-01-01"
                elif len(parts) == 2:
                    submission_str = f"{submission_str}-01"
            try:
                submission_date = pd.to_datetime(submission_str) if submission_str else None
            except (ValueError, TypeError):
                submission_date = None

            parts = collection_str.split("-")
            if len(parts) == 1:
                collection_str = f"{collection_str}-01-01"
            elif len(parts) == 2:
                collection_str = f"{collection_str}-01"
            try:
                collection_date = pd.to_datetime(collection_str)
            except (ValueError, TypeError):
                continue

            if submission_date is None:
                submission_date = collection_date
            if submission_date >= pd.to_datetime(cutoff_date):
                continue
            if collection_date.year >= 2010:
                row_data = {
                    "accession number": isolate_id,
                    "name": (
                        str(isolate_name_info[0])
                        if len(isolate_name_info) > 0 and pd.notna(isolate_name_info[0])
                        else ""
                    ),
                    "clade": (
                        clade_info[0]
                        if len(clade_info) > 0 and pd.notna(clade_info[0])
                        else ""
                    ),
                    "collection_date": collection_date.strftime("%Y-%m-%d"),
                    "submission_date": submission_date.strftime("%Y-%m-%d"),
                    "season": collection_date.year - 1
                    if collection_date.month < 2
                    else collection_date.year,
                }

                for j, aa in enumerate(seq):
                    row_data[f"X{j+1}"] = aa
                valid_rows.append(row_data)
    return valid_rows


# Clean and align FASTA sequences with their metadata table.
# 处理 FASTA 序列并与元数据表对齐。
def prepare_seq(
    input_fasta_seq: str | Path,
    info_file: str | Path,
    cutoff_date: str,
    subtype: str = "H3N2",
    processes: int | None = None,
) -> pd.DataFrame:
    sequences: List[str] = []
    names: List[str] = []
    for record in SeqIO.parse(str(input_fasta_seq), "fasta"):
        sequences.append(str(record.seq))
        names.append(str(record.id))

    info_path = Path(info_file)
    if info_path.suffix.lower() == ".csv":
        info_data = pd.read_csv(info_path)
    else:
        info_data = pd.read_excel(info_path)

    n_cores = processes if processes is not None else cpu_count()
    chunk_size = len(sequences) // n_cores + 1
    sequence_chunks = [
        sequences[i : i + chunk_size] for i in range(0, len(sequences), chunk_size)
    ]
    name_chunks = [
        names[i : i + chunk_size] for i in range(0, len(names), chunk_size)
    ]
    chunks = [
        (seq_chunk, name_chunk, info_data)
        for seq_chunk, name_chunk in zip(sequence_chunks, name_chunks)
    ]

    process_func = partial(
        parallel_sequence_processing, cutoff_date=cutoff_date, subtype=subtype
    )

    if n_cores <= 1 or len(chunks) <= 1:
        results = [process_func(chunk) for chunk in chunks]
    else:
        with Pool(n_cores) as pool:
            results = pool.map(process_func, chunks)

    all_rows = list(itertools.chain(*results))
    return pd.DataFrame(all_rows)


# Generate the subtype-specific HA sequence wide table.
# 生成指定亚型的 HA 序列宽表。
def generate_sequence_table(
    subtype: str,
    fasta_path: Path,
    info_path: Path,
    cutoff_date: str,
    processes: int | None = None,
) -> pd.DataFrame:
    return prepare_seq(
        input_fasta_seq=fasta_path,
        info_file=info_path,
        cutoff_date=cutoff_date,
        subtype=subtype,
        processes=processes,
    )


def prepare_all_subtypes(
    subtype_to_paths: dict,
    cutoff_date: str,
) -> dict:
    
    results: dict = {}
    for subtype, (fasta_path, info_path) in subtype_to_paths.items():
        results[subtype] = generate_sequence_table(
            subtype=subtype,
            fasta_path=fasta_path,
            info_path=info_path,
            cutoff_date=cutoff_date,
        )
    return results


__all__ = [
    "generate_sequence_table",
    "prepare_all_subtypes",
]
