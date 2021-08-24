# Task 2 Ranking

This is an implementation of the ranking method described on the [challenge website](https://fets-ai.github.io/Challenge/participate/#task-2-evaluation-details). To run this on your computer, you need to install R and the challengeR toolkit, as described in their [repository](https://github.com/wiesenfa/challengeR/#installation). The script `compute_ranking.R` should be invoked by
```
Rscript compute_ranking.R data_path [report_save_dir]
```
and takes two positional arguments as input: 
- `data_path` specifies the path to the directory that contains yaml-files with the evaluation results (there will be one for each testing institution in the federated evaluation).
- `report_save_dir` (optional) specifies the path to the directory where ranking analysis reports should be saved to. If not present, no reports are created.
