from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    DICE_METRICS,
    HAUSD_METRICS,
    NICE_METRIC_MAPPING,
    REGION_LIST,
    get_figsize,
    plt_save_and_close,
)


FETS_MODELS = ["8", "10", "11", "12", "54"]
FETS_MODELS_RANKING_ORDER = ["10", "11", "54", "12", "8"]
DATASETS_FOR_RANKING = [
    "4",
    "6",
    "7",
    "8",
    "9",
    "10",
    "12",
    "13",
    "14",
    "15",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "34",
    "35",
]


def aggregate_per_site(metric_df, agg_fn="mean", metrics=None):
    """Aggregate all metric values according to agg_fn across each dataset and model."""
    if metrics is None:
        metrics = DICE_METRICS + HAUSD_METRICS
    metric_df = metric_df.copy()
    if "dataset" not in metric_df.columns:
        print(
            "WARNING: Couldn't find a `dataset` column over which to aggregate. Inserting dummy..."
        )
        metric_df["dataset"] = "dummy"
    invert_metrics = list(set(HAUSD_METRICS).intersection(metrics))
    metric_df.loc[:, invert_metrics] *= -1  # For quantiles
    per_site_agg_results = metric_df.groupby(["model", "dataset"])[metrics].agg(agg_fn)
    per_site_agg_results.loc[:, invert_metrics] *= -1
    metric_df.loc[:, invert_metrics] *= -1  # Do not change metric_df in this function
    return per_site_agg_results.reset_index()


def add_regionsize_columns(data: pd.DataFrame, inplace=False):
    if inplace:
        extended_data = data
    else:
        extended_data = data.copy()
    for region in REGION_LIST:
        extended_data[f"size_{region}"] = (
            extended_data[f"TP_{region}"] + extended_data[f"FN_{region}"]
        )
        extended_data[f"size_pred_{region}"] = (
            extended_data[f"TP_{region}"] + extended_data[f"FP_{region}"]
        )
        if "missing_pred" in extended_data.columns:
            extended_data.loc[
                extended_data.missing_pred, [f"size_{region}", f"size_pred_{region}"]
            ] = np.nan
    return extended_data


def overview_plot_heatmap_single(
    data: pd.DataFrame,
    metric: str,
    model_col="model",
    site_col="dataset",
    vmin: float = None,
    vmax: float = None,
    cmap=None,
    model_order=None,
    highlight_models=None,
    site_order=None,
    site_sample_sizes=None,
    fig_kwargs=None,
    sns_kwargs=None,
):
    if fig_kwargs is None:
        fig_kwargs = {}
    if sns_kwargs is None:
        sns_kwargs = {}

    metric_label = NICE_METRIC_MAPPING[metric]
    fig = plt.figure(**fig_kwargs)

    # order by mean metric value.
    # missing value imputation: median per dataset.
    # Note: This is just for ordering models; missing values are still shown as missing in the heatmap!
    tmp_filled_na_df = data.copy().reset_index()
    median_per_ds = tmp_filled_na_df.groupby("dataset")[metric].median()
    for ds in tmp_filled_na_df.dataset.unique():
        tmp_filled_na_df.loc[
            tmp_filled_na_df[metric].isna() & (tmp_filled_na_df.dataset == ds), metric
        ] = median_per_ds[ds]
    mean_per_model = tmp_filled_na_df.groupby("model")[metric].mean()
    model_order = mean_per_model.sort_values(ascending="Hausd" in metric).index.tolist()

    if site_order is None:
        site_order = data[site_col].unique().tolist()
    if model_order is None:
        model_order = data[model_col].unique().tolist()
    plot_data = data.pivot(index=model_col, columns=site_col, values=metric)
    plot_data.index = plot_data.index.astype("string")
    plot_data = plot_data.loc[model_order]
    plot_data = plot_data.loc[:, site_order]
    # empty | model mean | empty
    # dataset size | heatmap | colorbar
    w, h = fig.get_figwidth(), fig.get_figheight()
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.3, 1, 0.05],
        height_ratios=[0.3 * w / h, 1 * w / h],
        hspace=0.05,
        wspace=0.275,
    )
    mean_model_ax = fig.add_subplot(grid[0, 1])
    ds_ax = fig.add_subplot(grid[1, 0])
    heatmap_ax = fig.add_subplot(grid[1, 1])
    cbar_ax = fig.add_subplot(grid[1, 2])
    sns.heatmap(
        data=plot_data.transpose(),
        ax=heatmap_ax,
        cbar_ax=cbar_ax,
        cbar_kws={"label": metric_label},
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **sns_kwargs,
    )
    # ax.set_title(metric)
    yticklabs = []
    heatmap_ax.set_yticks(np.arange(len(site_order)) + 0.5, labels=yticklabs)
    heatmap_ax.set_ylabel("")
    heatmap_ax.set_xlabel("Model ID")
    if highlight_models:
        # Some custom ticks for the x-axis
        new_ticks = []
        new_ticklabels = []
        for model_idx, model in enumerate(model_order):
            if model in highlight_models:
                new_ticks.append(model_idx + 0.5)
                # new_ticklabels.append(" ")
                new_ticklabels.append(model)
        heatmap_ax.set_xticks(new_ticks, labels=new_ticklabels)
        # rotate the tick labels for the x axis by 90 degrees
        heatmap_ax.tick_params(axis="x", rotation=90)
    sns.barplot(
        x=site_sample_sizes,
        y=site_order,
        color="tab:gray",
        ax=ds_ax,
    )
    ds_ax.set_ylabel("Institution ID")
    ds_ax.set_yticks(np.arange(len(site_order)), labels=site_order)
    # Move y-ticks to the right side
    ds_ax.yaxis.tick_right()
    # ds_ax.yaxis.set_label_position("right")
    ds_ax.invert_xaxis()
    ds_ax.set_xlabel("#test cases")
    ds_ax.set_title(" ")

    # mean model
    if mean_model_ax is not None:
        sns.barplot(
            data=mean_per_model,
            order=model_order,
            color="tab:gray",
            ax=mean_model_ax,
        )
        mean_model_ax.set_ylabel(metric_label)
        mean_model_ax.set_xlabel("")
        mean_model_ax.set_ylim(vmin, vmax)
        mean_model_ax.set_xticks(
            np.array(new_ticks) - 0.5, labels=new_ticklabels
        )  # IDK why -0.5

    return fig, plot_data


def plot_results_overview_single_model(
    metrics_df: pd.DataFrame, model_id: str, metric: str, output_file: Path
):
    test_scores = metrics_df.copy()
    test_scores["Dice_mean"] = test_scores[DICE_METRICS].mean(axis=1)
    test_scores["Hausdorff95_mean"] = test_scores[HAUSD_METRICS].mean(axis=1)

    def compute_num_per_site(df):
        return (
            df.groupby("dataset")
            .case_id.agg(lambda s: len(s.unique()))
            .sort_values(ascending=False)
        )

    def filter_df(
        scores_df, metric_region, top_models, region_thresh=None, ds_thresh=None
    ):
        if not isinstance(top_models, (list, tuple)):
            top_models = [top_models]
        scores_df = scores_df.copy()
        region = metric_region.split("_")[1]

        # filter small regions
        if region_thresh is not None:
            scores_df = add_regionsize_columns(scores_df)
            # print(compute_num_per_site(scores_df).sum())
            if region == "mean":
                scores_df = scores_df.loc[
                    (scores_df["size_WT"] >= region_thresh)
                    & (scores_df["size_TC"] >= region_thresh)
                    & (scores_df["size_ET"] >= region_thresh),
                    :,
                ]
            else:
                scores_df = scores_df[scores_df[f"size_{region}"] >= region_thresh]
            # print(compute_num_per_site(scores_df).sum())
        # filter small sites
        if ds_thresh is not None:
            site_sizes = compute_num_per_site(scores_df)
            filter_datasets = site_sizes[site_sizes >= ds_thresh].index.to_list()
            scores_df = scores_df[scores_df.dataset.isin(filter_datasets)]
            # print(compute_num_per_site(scores_df).sum())
        top_results_test = scores_df.loc[
            scores_df.model.isin(top_models),
            ["dataset", "case_id", "model", metric_region],
        ]
        return top_results_test

    test_scores_filtered = filter_df(
        test_scores,
        metric,
        model_id,
        region_thresh=500,
        ds_thresh=40,
    )
    ds_order = (
        test_scores_filtered.groupby("dataset")[metric]
        .median()
        .sort_values()
        .index.tolist()
    )
    fig, ax = plt.subplots(
        figsize=get_figsize(textwidth_factor=0.5), layout="constrained"
    )
    fig.patch.set_facecolor("none")
    sns.stripplot(
        test_scores_filtered,
        x="dataset",
        y=metric,
        order=ds_order,
        color="tab:red",
        size=1.5,
        ax=ax,
    )
    ax.set_ylabel(NICE_METRIC_MAPPING[metric])
    ax.set_xlabel("Test data site", labelpad=0)
    ax.set_xticks(ticks=ax.get_xticks(), labels=[])
    plt_save_and_close(fig, output_file)


def plot_results_mean_per_dataset(
    metrics_df: pd.DataFrame,
    included_models_ranking_order,
    output_file: Path,
    sort_by_size=True,
    metric="Dice_ET",
    cmap="magma",
):
    plot_data = metrics_df.copy()
    if metric == "Dice_mean" and metric not in plot_data.columns:
        plot_data["Dice_mean"] = plot_data[DICE_METRICS].mean(axis=1)
    site_sizes = plot_data.groupby("dataset").case_id.agg(lambda s: len(s.unique()))
    site_order = sorted(plot_data.dataset.unique().tolist())
    if sort_by_size:
        site_order = site_sizes.sort_values(ascending=False).index.to_list()
    site_sizes = site_sizes[site_order]
    # # Example for a different aggregation than mean
    # agg_fn = lambda s: np.percentile(s, q=10)
    per_site_agg_results = aggregate_per_site(
        plot_data, agg_fn="mean", metrics=[metric]
    )
    # remove models that are not included
    per_site_agg_results = per_site_agg_results.loc[
        per_site_agg_results.model.isin(included_models_ranking_order)
    ]
    if "Hausd" in metric:
        cmap = sns.color_palette(cmap + "_r", as_cmap=True)
    else:
        cmap = sns.color_palette(cmap, as_cmap=True)
    fig, plot_data = overview_plot_heatmap_single(
        per_site_agg_results,
        metric=metric,
        model_order=included_models_ranking_order,
        site_order=site_order,
        site_sample_sizes=site_sizes.to_numpy(),
        cmap=cmap,
        vmin=0.5 if "Dice" in metric else 0,
        vmax=1.0 if "Dice" in metric else 80,
        fig_kwargs={"figsize": get_figsize(aspect_ratio=0.7)},
        highlight_models=["8", "10", "11", "12", "54"],
    )
    plt_save_and_close(fig, output_file)


def plot_brats_vs_fets_testset_results_single_metric_nosize(
    all_metrics_df: pd.DataFrame,
    model_list: list[str],
    output_file: Path,
    metric="Dice_ET",
    distinguish_brats_fets=False,
):
    # model_list: Choose for which models the metric distribution should be visualized
    brats_subset = {
        "27": 157,  # Collaborator11
        "28": 130,  # Collaborator12
        "29": 26,  # Collaborator13
        "30": 52,  # Collaborator14
        "31": 11,  # Collaborator15
        "32": 9,  # Collaborator16
        # unseen during training
        "21": 28,  # Collaborator7
        "22": 124,  # Collaborator8
    }
    brats_ood_collabs = ["21", "22"]
    # pool multiple models together (if len(model_list) > 1)
    plot_data = all_metrics_df[all_metrics_df.model.isin(model_list)].copy()
    if metric == "Dice_mean":
        plot_data["Dice_mean"] = plot_data[DICE_METRICS].mean(axis=1)
    plot_data.loc[:, plot_data.columns.isin(HAUSD_METRICS)] = plot_data.loc[
        :, plot_data.columns.isin(HAUSD_METRICS)
    ].clip(lower=0, upper=100)
    if distinguish_brats_fets:
        plot_data["origin"] = "FeTS"
        plot_data.loc[plot_data.dataset.isin(list(brats_subset)), "origin"] = "BraTS"
        plot_data.loc[plot_data.dataset.isin(brats_ood_collabs), "origin"] = (
            "BraTS unseen"
        )
        hue_order = ["BraTS", "BraTS unseen", "FeTS"]
    else:
        plot_data["origin"] = "Unseen during training"
        plot_data.loc[plot_data.dataset.isin(list(brats_subset)), "origin"] = (
            "Seen during training"
        )
        plot_data.loc[plot_data.dataset.isin(brats_ood_collabs), "origin"] = (
            "Unseen during training"
        )
        hue_order = ["Seen during training", "Unseen during training"]
    # Visualize the complete distribution of test cases (BraTS vs. FeTS), for each institutions
    # order datasets by median Dice
    ds_order = (
        plot_data.groupby("dataset")[metric].median().sort_values().index.tolist()
    )
    fig, ax = plt.subplots(figsize=get_figsize(aspect_ratio=0.5))
    metric_label = NICE_METRIC_MAPPING[metric]
    # red shades to match the fig 1 colors
    color_list = (
        ["#aa676a", "#ff957f", "#d62728"]
        if distinguish_brats_fets
        else ["#ff957f", "#d62728"]
    )
    sns.boxplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="origin",
        dodge=False,
        fliersize=0.0,
        linewidth=1,
        ax=ax,
        order=ds_order,
        hue_order=hue_order,
        fill=False,
        palette=color_list,
        boxprops=dict(alpha=0.8),
    )
    # plot all stripplot points in the same color (gray)
    sns.stripplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="origin",
        dodge=False,
        ax=ax,
        order=ds_order,
        hue_order=hue_order,
        palette=["gray"] * len(color_list),
        size=1.5,
        alpha=0.7,
        jitter=0.2,
        zorder=0,
    )
    # remove legend entries of stripplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(hue_order)],
        labels[: len(hue_order)],
        title="Source" if distinguish_brats_fets else None,
    )
    # rotate xticklabels
    _ = ax.set_xticks(ax.get_xticks(), ds_order)
    ax.set_xlabel("Institution ID")
    ax.set_ylabel(metric_label)

    # Add "n =" labels
    ypos = ax.get_ylim()[1] * 1.01
    ax.text(
        -0.5, ypos, "n =", fontsize=5, ha="right", va="bottom", fontweight="bold", color="gray"
    )

    # Add individual sample counts above bars
    sample_counts = plot_data.groupby("dataset").size().to_dict()
    for i, cat in enumerate(ds_order):
        ax.text(
            i, ypos, f"{sample_counts[cat]}", fontsize=5, ha="center", va="bottom", color="gray"
        )
    if output_file is not None:
        plt_save_and_close(fig, output_file)


def rank_then_aggregate(
    df, agg="mean", combine_metrics=False, ties="min", treat_na="bottom"
):
    # dataframe columns: case_id, model, metric, value (=> long form!)
    # NOTE assumes higher values are better. Invert if necessary before calling this function.
    # rank the models for each case and metric
    assert df.duplicated(subset=["case_id", "model", "metric"]).sum() == 0
    df["rank"] = df.groupby(["case_id", "metric"])["value"].rank(
        ascending=False, method=ties, na_option=treat_na
    )
    # aggregate the ranks for each model (and metric if combine_metrics -> BraTS style)
    if combine_metrics:
        agg_ranks = df.groupby(["model"])["rank"].agg(agg).reset_index()
        agg_ranks.rename(columns={"rank": f"{agg}_rank"}, inplace=True)
        agg_ranks["rank"] = agg_ranks[f"{agg}_rank"].rank(method=ties)
    else:
        agg_ranks = df.groupby(["model", "metric"])["rank"].agg(agg).reset_index()
        agg_ranks = agg_ranks.rename(columns={"rank": f"{agg}_rank"})
        agg_ranks["rank"] = agg_ranks.groupby("metric")[f"{agg}_rank"].rank(method=ties)
    return agg_ranks, df.loc[:, ["case_id", "model", "metric", "rank"]].copy()


def compute_fets_ranking_rankthenmean(metric_results: pd.DataFrame):
    metric_results = metric_results[
        ["dataset", "case_id", "model"] + DICE_METRICS + HAUSD_METRICS
    ]
    # assert that there are no duplicates (ie multiple metrics for the same model-case-dataset)
    if metric_results.duplicated(subset=["dataset", "case_id", "model"]).any():
        raise ValueError("There are duplicates in the metric results")
    metric_results = metric_results.melt(
        id_vars=["dataset", "case_id", "model"],
        var_name="metric",
        value_name="value",
    )
    # negate hausdorff distance values -> higher is better
    metric_results["value"] = metric_results["value"] * metric_results["metric"].apply(
        lambda x: -1 if "hausdorff" in x.lower() else 1
    )
    all_institution_rankings = []
    case_rankings = []
    # compute rankings for each institution and region-metric
    for ds_id, single_institution_df in metric_results.groupby("dataset"):
        single_institution_ranking, case_ranks = rank_then_aggregate(
            single_institution_df, agg="mean", combine_metrics=False
        )
        # this has one rank for each model and metric (or just for each model if brats_style=True)
        single_institution_ranking["dataset"] = ds_id
        case_ranks["dataset"] = ds_id
        all_institution_rankings.append(single_institution_ranking)
        case_rankings.append(case_ranks)
    all_institution_rankings = pd.concat(all_institution_rankings, ignore_index=True)
    case_rankings = pd.concat(case_rankings, ignore_index=True)
    # average over all institutions and region-metrics
    final_ranking = all_institution_rankings.groupby("model")["rank"].mean()
    final_ranking = final_ranking.rename("mean_rank").reset_index()
    final_ranking["final_rank"] = final_ranking["mean_rank"].rank(method="min")
    final_ranking = final_ranking.sort_values("final_rank")
    return final_ranking, all_institution_rankings, case_rankings


def analysis_task2(results_df: pd.DataFrame, figures_dir: Path):
    # Compute ranking
    # for the extended ranking, the subset of sites (= datasets) is used, on which all models could be evaluated:
    tmp_for_ranking_df = results_df[results_df.dataset.isin(DATASETS_FOR_RANKING)]
    final_rankings, _, _ = compute_fets_ranking_rankthenmean(tmp_for_ranking_df)
    # # for the official MICCAI ranking, use this:
    # tmp_for_ranking_df = results_df[results_df["mode"].isin(FETS_MODELS)]
    # official_ranking, _, _ = compute_fets_ranking_rankthenmean(tmp_for_ranking_df)

    ranked_model_list = final_rankings.sort_values("final_rank")["model"].to_list()
    print(f"Ranked models: {ranked_model_list}")
    # Fig. 1 (strip plot)
    # use axis label size 7, tick label size 5
    with matplotlib.rc_context(
        {
            "xtick.labelsize": 1,
            "ytick.labelsize": 5,
            "axes.labelsize": 7,
        }
    ):
        plot_results_overview_single_model(
            results_df,
            model_id=ranked_model_list[0],
            metric="Dice_mean",
            output_file=figures_dir / "fig1_results_task2",
        )

    # # Fig. 2: Task 2 raw results
    # (and all other metrics/regions for appendix)
    for metric in DICE_METRICS + HAUSD_METRICS + ["Dice_mean"]:
        if metric == "Dice_mean":
            outfile = figures_dir / f"fig2_{metric}"
        else:
            # extra plots for individual tumor region metrics
            outfile = figures_dir / f"suppl-fig_results_{metric}"
        plot_results_mean_per_dataset(
            results_df,
            [x for x in ranked_model_list if x != "53"],  # model 53 is broken
            outfile,
            sort_by_size=True,
            metric=metric,
        )

    # Fig. 3
    plot_brats_vs_fets_testset_results_single_metric_nosize(
        results_df,
        [ranked_model_list[0]],
        figures_dir / "fig3_mean_dice",
        metric="Dice_mean",
    )
    # some extra plots for individual tumor regions
    for metric in DICE_METRICS:
        plot_brats_vs_fets_testset_results_single_metric_nosize(
            results_df,
            [ranked_model_list[0]],
            figures_dir / f"suppl-fig_best_model_results_{metric}",
            metric=metric,
        )
