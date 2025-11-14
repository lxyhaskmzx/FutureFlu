# FutureFlu Pipeline（中文说明）

## 致谢
我们谨此感谢所有数据贡献者：包括负责采集样本的作者及其原始实验室，以及负责测序并分享序列与元数据的提交实验室。本项目的分析基于通过 GISAID 平台共享的数据。

## 数据获取
本仓库不再分发任何 GISAID 序列或元数据。根据 GISAID 使用条款，请用户使用自身账号直接从 GISAID 获取数据。本仓库仅提供可在本地对 GISAID 下载文件运行的脚本与流程。

## 概述
FutureFlu 流程整合了序列准备、突变预测与适应度计算，覆盖 A(H1N1)pdm09、A(H3N2)、B/Victoria。运行后会在 `outputs/<subtype>_<hemisphere>/season_<year>/` 下生成中间结果，包含 2013–2024 每个季节的突变预测、theta 拟合、流行度与谱系适应度表。

## 输入
- FASTA 与 metadata 放在 `data/dataset/`；使用 `data/dataset/metadata/demo_metadata.xlsx` 从 GISAID 下载相应的 metadata CSV 与 HA FASTA。
- Positivity 率：`data/positivity/*.csv`
- EVEscape 数据：`data/EVEscape/<Subtype>_evescape/`

## 环境
- Python 3.11，并执行 `pip install -r requirements.txt`

## 运行方式
### 批量运行
```bash
python scripts/run_full_pipeline.py --jobs 6
```
读取默认配置，覆盖 A(H1N1)pdm09、A(H3N2)、B/Victoria，季节 2013–2024。

### 单独运行
```bash
python scripts/run_pipeline.py --config configs/h1n1_pre2024.json
```
加载 `configs/h1n1_pre2024.json`；可在该文件中修改数据路径与 theta。

## 输出
每个季节的 CSV 文件位于 `outputs/<subtype>_<hemisphere>/season_<year>/`（如 `fitness.csv`、`theta_fitting.csv`、`mutation_prediction.csv`、`prevalence_result.csv`）。
