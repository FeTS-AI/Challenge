from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    get_figsize,
    plt_save_and_close,
    NICE_METRIC_MAPPING,
    DICE_METRICS,
    HAUSD_METRICS,
)


TEAMS = [
    "Flair",
    "FLSTAR",
    "gauravsingh",
    "HTTUAS",
    "rigg",
    "RoFL",
    "Sanctuary",
]


def compute_ranking(all_data, convscores, convscore_weight=3, tie_resolver="average"):
    # Compute the ranking from above data
    tmp_all_data = all_data.copy()
    tmp_all_data.set_index(["case_idx", "team"], inplace=True)
    rank_df_dice = tmp_all_data.groupby("case_idx")[DICE_METRICS].rank(
        ascending=False, method=tie_resolver
    )
    rank_df_hd = tmp_all_data.groupby("case_idx")[HAUSD_METRICS].rank(
        ascending=True, method=tie_resolver
    )
    rank_df = pd.concat([rank_df_dice, rank_df_hd], axis=1)
    rank_df.rename(
        columns={x: x + "_rank" for x in DICE_METRICS + HAUSD_METRICS}, inplace=True
    )
    ranking_cols = [x + "_rank" for x in DICE_METRICS + HAUSD_METRICS]
    rank_df = rank_df.reset_index()

    # plot the ranking distribution (only dice and hausdorff)
    rank_df["cum_rank"] = rank_df[ranking_cols].sum(axis=1) / len(ranking_cols)
    mean_ranks = rank_df.groupby("team")["cum_rank"].mean()

    # add communication efficiency ranks. Note: only add them for the teams, not baselines
    teams_comm_data = convscores[convscores["team"].isin(TEAMS)].copy()
    # Note: ascending=False because I assume projected DSC over time AUC => higher is better
    teams_comm_data["rank"] = teams_comm_data["communication_metric"].rank(
        ascending=False, method=tie_resolver
    )
    for w in range(1, convscore_weight + 1):
        to_merge_df = teams_comm_data[["team", "rank"]].rename(
            columns={"rank": f"comm{w}_rank"}
        )
        ranking_cols.append(f"comm{w}_rank")
        rank_df = rank_df.merge(to_merge_df, on="team", how="left")
    rank_df["cum_rank_plus_comm"] = rank_df[ranking_cols].sum(axis=1) / len(
        ranking_cols
    )
    mean_ranks = rank_df.groupby("team")["cum_rank_plus_comm"].mean()
    return rank_df, mean_ranks


def plot_violin_figure_task1(
    data, metric="metric", ax=None, map_team_names=None, y_order=None
):
    data = data.copy()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=get_figsize(), layout="constrained")
    if map_team_names:
        data["team"] = data["team"].map(map_team_names)
        y_order = list(map(lambda k: map_team_names[k], y_order))
    if y_order is None:
        y_order = sorted(data.team.unique())
    sns.violinplot(
        data=data,
        x=metric,
        y="team",
        hue="team",
        inner=None,
        dodge=False,
        order=y_order,
        hue_order=y_order,
        density_norm="count",
        # palette=my_pal,  # default palette is ok for <= 10 teams
        cut=0,
        gridsize=1000,
        ax=ax,
        linewidth=1,
    )
    sns.boxplot(
        data=data,
        x=metric,
        y="team",
        hue="team",
        dodge=False,
        order=y_order,
        hue_order=y_order,
        # palette=my_pal,
        linewidth=1,
        width=0.2,
        showmeans=True,
        meanprops=dict(
            marker="x", markeredgecolor="red", markersize=6, markeredgewidth=2
        ),
        medianprops=dict(color="w", linewidth=2),
        flierprops=dict(marker="o", markersize=3),
        ax=ax,
        zorder=2,
    )
    # hide legend
    if metric in NICE_METRIC_MAPPING:
        ax.set_xlabel(NICE_METRIC_MAPPING[metric])
    else:
        ax.set_xlabel(metric)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_ylabel("")
    return fig, ax


def plot_convscore_task1(comm_data, output_dir, map_team_names=None, team_order=None):
    comm_data = comm_data.copy()
    fig, ax = plt.subplots(
        figsize=get_figsize(textwidth_factor=0.5), layout="constrained"
    )
    order = None
    if team_order is not None:
        order = team_order
    if map_team_names:
        comm_data["team"] = comm_data["team"].map(map_team_names)
        if order is not None:
            order = list(map(lambda k: map_team_names[k], order))
    sns.barplot(
        data=comm_data,
        x="team",
        y="communication_metric",
        hue="team",
        dodge=False,
        order=order,
        hue_order=order,
        ax=ax,
        # width=0.2,
    )
    ax.set_ylabel("Convergence score")
    ax.set_xlabel("")
    # rotate xticklabels
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    return fig, ax


def make_overview_table(
    metrics_df, convscores, baseline_metrics_df, mean_ranks, output_file: Path
):
    table_data = metrics_df.copy()
    teams_ranking_order = mean_ranks.sort_values().index.to_list()
    table_rows = teams_ranking_order + list(baseline_metrics_df.team.unique())
    # NOTE the case_idx might be inconsistent between the teams and baselines, but for mean it doesn't matter
    baseline_metrics_df["case_idx"] = baseline_metrics_df["case_id"].map(
        {
            case_id: i
            for i, case_id in enumerate(sorted(baseline_metrics_df["case_id"].unique()))
        }
    )
    table_data = pd.concat([table_data, baseline_metrics_df], axis=0, ignore_index=True)
    table_data = table_data.groupby("team").mean(numeric_only=True)

    table_data = pd.concat(
        [
            table_data,
            convscores.set_index("team"),
            mean_ranks.rename("Ranking score"),
        ],
        axis=1,
    )
    # sort by ranking
    table_data = table_data.loc[table_rows, :]
    # style the table for latex output: apply a color map for each column to highlight the best team
    table_data.loc[:, DICE_METRICS] *= 100
    table_data.loc[:, "communication_metric"] *= 100
    table_data = table_data.loc[
        :, DICE_METRICS + HAUSD_METRICS + ["communication_metric", "Ranking score"]
    ]
    table_data.rename(columns=NICE_METRIC_MAPPING, inplace=True)
    cm = sns.color_palette("magma", as_cmap=True)
    cm_rev = sns.color_palette("magma_r", as_cmap=True)
    s = table_data.style.background_gradient(
        cmap=cm,
        axis=0,
        subset=[NICE_METRIC_MAPPING[x] for x in DICE_METRICS],
    )
    s = s.background_gradient(
        cmap=cm_rev,
        axis=0,
        subset=[NICE_METRIC_MAPPING[x] for x in HAUSD_METRICS],
    )
    s = s.background_gradient(
        cmap=cm,
        axis=0,
        subset=(
            ~table_data.index.isin(["Centralized"]),
            NICE_METRIC_MAPPING["communication_metric"],
        ),
    )
    s = s.background_gradient(
        cmap=cm_rev,
        axis=0,
        subset=(
            ~table_data.index.isin(["Centralized", "Default"]),
            ["Rank score"],
        ),
    )
    # rename the columns
    s = s.format(
        "{:.1f}",
        na_rep="-",
        subset=(slice(None), ~table_data.columns.isin(["Rank score"])),
    )
    s = s.format(
        "{:.2f}",
        na_rep="-",
        subset=(slice(None), table_data.columns.isin(["Rank score"])),
    )
    if output_file.suffix != ".tex":
        raise ValueError("Only LaTeX output is supported for the table.")
    s.to_latex(
        output_file,
        hrules=True,
        convert_css=True,
        position_float="centering",
    )
    s.to_html(
        output_file.with_suffix(".html"),
        border=1,
    )


def analysis_task1(metrics_df, convscores, baseline_metrics_df, output_dir):
    # compute ranking
    ranking_df, mean_ranks = compute_ranking(metrics_df, convscores)
    teams_ranking_order = mean_ranks.sort_values().index.to_list()

    # All in one figure
    figname = "suppl-fig-4_allmetrics"
    fig, axes = plt.subplots(
        3,
        2,
        figsize=get_figsize(aspect_ratio=1.5),
        layout="constrained",
        sharey=True,
    )
    for i, metric in enumerate(DICE_METRICS + HAUSD_METRICS):
        curr_ax = axes[i % 3, i // 3]
        plot_violin_figure_task1(
            metrics_df,
            metric=metric,
            ax=curr_ax,
            y_order=teams_ranking_order,
        )
        if "dice" in metric.lower():
            curr_ax.set_xlim([-0.05, 1.05])
        else:
            curr_ax.set_xlim([-13, 388])
        if i % 3 != 2:
            curr_ax.set_xticklabels([])
    plt_save_and_close(fig, output_dir / f"{figname}.png")

    # Table with everything (dice, hausdorff, communication cost, ranking)
    make_overview_table(
        metrics_df,
        convscores,
        baseline_metrics_df,
        mean_ranks,
        output_dir / "suppl-table-1_overview_with_baselines.tex",
    )

    # Ranking plot
    figname = "suppl-fig-3_ranking_scores"
    fig, ax = plt.subplots(
        figsize=get_figsize(textwidth_factor=0.8), layout="constrained"
    )
    plot_violin_figure_task1(
        ranking_df, metric="cum_rank_plus_comm", y_order=teams_ranking_order, ax=ax
    )
    ax.set_xlabel("Ranking score")
    plt_save_and_close(fig, output_dir / f"{figname}.png")
    print("Final ranking (with communication):")
    print(mean_ranks.sort_values())
