import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from plots_task1 import analysis_task1
from plots_task2 import analysis_task2


def main():
    parser = argparse.ArgumentParser(description="Generate main figures")
    parser.add_argument("source_data_dir", type=str, help="Input file with results")
    parser.add_argument("output_dir", type=str, help="Output directory for figures")
    parser.add_argument(
        "--only-task1", action="store_true", help="Generate only task 1 figures"
    )
    parser.add_argument(
        "--only-task2", action="store_true", help="Generate only task 2 figures"
    )
    args = parser.parse_args()

    figures_dir = Path(args.output_dir)
    figures_dir.mkdir(exist_ok=True)
    source_data_dir = Path(args.source_data_dir)
    if not source_data_dir.exists() or not source_data_dir.is_dir():
        raise ValueError(
            f"Source data directory {source_data_dir} does not exist or is not a directory."
        )

    matplotlib.rcParams["font.size"] = 5
    matplotlib.rcParams["font.family"] = "Liberation Sans"

    if not args.only_task2:
        # TASK 1
        print("TASK 1")
        print("Loading data")
        task1_results = pd.read_csv(source_data_dir / "task1_metrics.csv")
        # the case IDs for the baseline results are different, therefore separate files from main results
        task1_baseline_results = pd.read_csv(
            source_data_dir / "task1_metrics_baselines.csv"
        )
        task1_convscores = pd.read_csv(source_data_dir / "task1_convscores.csv")
        task1_dir = figures_dir / "task1"
        task1_dir.mkdir(exist_ok=True, parents=True)
        print("Running analysis")
        analysis_task1(
            task1_results, task1_convscores, task1_baseline_results, task1_dir
        )

    if not args.only_task1:
        # TASK 2
        print("TASK 2")
        print("Loading data")
        # task2_results = pd.read_excel(source_data_file, sheet_name="Task2")
        task2_results = pd.read_csv(source_data_dir / "task2_results.csv")
        task2_dir = figures_dir / "task2"
        task2_dir.mkdir(exist_ok=True, parents=True)

        # use string identifiers for model and datasets
        task2_results["model"] = task2_results["model"].astype(str)
        task2_results["dataset"] = task2_results["dataset"].astype(str)
        print("Running analysis")
        analysis_task2(task2_results, task2_dir)


if __name__ == "__main__":
    main()
