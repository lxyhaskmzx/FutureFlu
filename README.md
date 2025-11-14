# FutureFlu Pipeline

[中文说明](README_zh.md)

## Acknowledgements
We gratefully acknowledge all data contributors, i.e., the Authors and their Originating laboratories responsible for obtaining the specimens, and their Submitting laboratories for generating the genetic sequence and metadata and sharing via the GISAID Initiative, on which this research is based.

## Data Access
This repository does not redistribute GISAID sequence data or metadata. Per GISAID’s Terms of Use, please obtain data directly from GISAID with your own account. We provide scripts/workflows that operate on files you download from GISAID.

## Overview
FutureFlu integrates sequence preparation, mutation prediction, and fitness scoring for seasonal influenza lineages A(H1N1)pdm09, A(H3N2), and B/Victoria. Running the workflow writes season-specific tables under `outputs/<subtype>_<hemisphere>/season_<year>/`, including per-clade `fitness.csv`, mutation predictions, theta fits, and prevalence summaries for 2013–2024.

## Inputs & Environment
- Place FASTA and metadata under `data/dataset/` using your GISAID downloads (metadata CSVs and HA FASTA exports). Use `data/dataset/metadata/demo_metadata.xlsx` with GISAID EpiFlu to download the corresponding metadata CSVs and HA FASTA files.
- Positivity curves in `data/positivity/*.csv`
- EVEscape summaries in `data/EVEscape/<Subtype>_evescape/`
- Use Python 3.11; install dependencies with `pip install -r requirements.txt`.

## Running the Pipeline
### Batch run
```bash
python scripts/run_full_pipeline.py --jobs 6
```
Runs default configs covering A(H1N1)pdm09, A(H3N2), and B/Victoria for seasons 2013–2024.

### Single run
```bash
python scripts/run_pipeline.py --config configs/h1n1_pre2024.json
```
Loads `configs/h1n1_pre2024.json`. Edit paths and theta in that file as needed.

## Outputs
Per-season CSV files are written to `outputs/<subtype>_<hemisphere>/season_<year>/` (e.g., `fitness.csv`, `theta_fitting.csv`, `mutation_prediction.csv`, `prevalence_result.csv`).
