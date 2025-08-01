"""
In this script we include a function that plots a boxplot and performs statistical
testing.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from statsmodels.formula.api import ols
import statsmodels.api as sm


def boxplot_with_stats(
        data, group_col, value_col, method="auto", ax=None, title="", **plot_kwargs
):
    """
    Plots boxplots of a continuous variable grouped by a categorical variable,
    performs statistical tests, and annotates significant differences.

    Parameters:
    - df: pandas DataFrame
    - group_col: column name (str) for group labels
    - value_col: column name (str) for continuous variable
    - method: 'auto', 'anova', or 'kruskal'. Determines the global test.
    - plot_kwargs: additional parameters to pass as kwargs to the
        violin_box_scatter_plot function such as palette, xlabel, ylabel, etc.
    """
    unique_groups = data[group_col].dropna().unique()
    data = data[
        data[value_col].notna()
    ]  # We drop na values in value_col to avoid nan stats
    data = data[
        data[group_col].notna()
    ]  # We also drop na values in the group_col for stats
    num_groups = len(unique_groups)

    if ax is None:
        # Plot boxplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    violin_box_scatter_plot(data, x=group_col, y=value_col, hue=group_col, ax=ax,
                            **plot_kwargs)

    # Global test
    if method == "auto":
        method = "anova" if num_groups <= 5 else "kruskal"

    if method == "anova":
        model = ols(f"{value_col} ~ C({group_col})", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        pval = anova_table["PR(>F)"][0]
        print("ANOVA result:\n", anova_table)
    elif method == "kruskal":
        groups = [
            group[value_col].dropna().values for name, group in data.groupby(group_col)
        ]
        stat, pval = kruskal(*groups)
        print(f"Kruskal-Wallis test statistic: {stat:.3f}, p-value: {pval:.4f}")
    else:
        raise ValueError("Invalid method. Choose from 'auto', 'anova', or 'kruskal'.")
    title = f"{title}Global Test Anova pval={round(pval, 5)}"
    ax.set_title(title)
    # Posthoc tests if global test is significant
    if pval < 0.05:
        if method == "anova":
            posthoc = pairwise_tukeyhsd(data[value_col], data[group_col])
            print("\nPosthoc (Tukey HSD):\n", posthoc.summary())
            sig_pairs = [
                (x[0], x[1], x[4])
                for x in posthoc._results_table.data[1:]
                if x[4] < 0.05
            ]
        elif method == "kruskal":
            posthoc = sp.posthoc_dunn(
                data, val_col=value_col, group_col=group_col, p_adjust="bonferroni"
            )
            print("\nPosthoc (Dunn test):\n", posthoc)
            sig_pairs = [
                (i, j, posthoc.loc[i, j])
                for i in posthoc.columns
                for j in posthoc.columns
                if i != j and posthoc.loc[i, j] < 0.05
            ]
            sig_pairs = list(
                {(min(a, b), max(a, b)): p for a, b, p in sig_pairs}.items()
            )  # remove duplicates

        # Annotate plot
        y_max = data[value_col].max()
        step = (y_max - data[value_col].min()) * 0.1
        height = y_max + step

        for pair in sig_pairs:
            group1, group2 = pair[0] if method == "kruskal" else pair[:2]
            pval = pair[1] if method == "kruskal" else pair[2]
            x1, x2 = (
                unique_groups.tolist().index(group1),
                unique_groups.tolist().index(group2),
            )
            ax.plot(
                [x1, x1, x2, x2],
                [height, height + 0.01, height + 0.01, height],
                lw=1.5,
                color="k",
            )
            text = (
                "***"
                if pval < 0.001
                else "**"
                if pval < 0.01
                else "*"
                if pval < 0.05
                else "ns"
            )
            ax.text(
                (x1 + x2) * 0.5,
                height + 0.015,
                text,
                ha="center",
                va="bottom",
                color="k",
            )
            height += step

    else:
        print("No significant difference found in the global test.")


def violin_box_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    order: list = None,
    palette: str or dict = "colorblind",
    violin_kwargs: dict = None,
    box_kwargs: dict = None,
    strip_kwargs: dict = None,
    figsize: tuple = (10, 6),
    title: str = "",
    xlabel: str = None,
    ylabel: str = None,
    show_legend: bool = True,
    save_path: str = None,
    df_stats=None,
    ax=None,
):
    """
    Plots a violin plot with inner boxplot and jittered scatter points from a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - x: categorical column (str)
    - y: numeric column (str)
    - hue: optional grouping variable (str)
    - order: custom order of categories on x-axis (list)
    - palette: color palette (str or dict)
    - violin_kwargs, box_kwargs, strip_kwargs: dicts for styling each layer
    - figsize: tuple, figure size
    - title: plot title
    - xlabel, ylabel: axis labels
    - show_legend: toggle legend
    - stats_df: pandas DataFrame containing the results from
        neuroblocks.stats.stat_tests.stat_tests_features. Can be the results from anova
        (if only two groups) or from posthoc tests if more than two groups.
    - save_path: path to save the figure (str)
    """

    # Sort by x values so that they keep same order at different boxplots
    # independently of the data order
    df = df.sort_values(by=x)

    sns.set(style="whitegrid")

    violin_kwargs = violin_kwargs or {
        "inner": None, "linewidth": 1, "alpha": 0.6
    }
    box_kwargs = box_kwargs or {
        "width": 0.1, "fliersize": 0, "linewidth": 1.2
    }
    strip_kwargs = strip_kwargs or {
        "jitter": 0.2, "size": 3, "alpha": 0.5, "linewidth": 0.3
    }

    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=figsize)

    # Layer 1: Violin plot
    sns.violinplot(
        data=df, x=x, y=y, hue=hue, order=order, ax=ax,
        palette=palette, **violin_kwargs
    )

    # Layer 2: Boxplot
    sns.boxplot(
        data=df, x=x, y=y, hue=hue, order=order, ax=ax,
        palette=palette, showcaps=True,
        whiskerprops={'linewidth':1.2}, **box_kwargs
    )

    # Layer 3: Scatter points
    sns.stripplot(
        data=df, x=x, y=y, hue=hue, order=order, ax=ax,
        palette=palette, dodge=True if hue else False, **strip_kwargs
    )

    # Title and labels
    ax.set_title(title, fontsize=14)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # Add statistics if stats_df is passed
    if df_stats is not None:
        add_stat_annotation(ax, df, x, y, df_stats, y, order=order)

    # Handle legend
    if hue and show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend(handles[0:len(set(df[hue]))], labels[0:len(set(df[hue]))])
    else:
        ax.legend([], [], frameon=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    # Return figure and axis if created within the function
    if return_fig:
        return fig, ax
    else:
        return ax


def add_stat_annotation(
        ax, df, x, y, posthoc_df, feature, order=None, test_name=None, y_offset=0.05
):
    """
    Add statistical annotation to a plot from posthoc_df.

    Parameters:
    - ax: matplotlib axis to annotate
    - df: dataframe used for the plot
    - x, y: column names used for plotting
    - posthoc_df: output from stat_tests_features (posthoc results)
    - feature: which feature's comparisons to annotate
    - order: custom x-axis order
    - test_name: optional, string to display above significance bars
    - y_offset: fraction of max(y) to use for spacing annotations vertically
    """
    if posthoc_df is None or posthoc_df.empty:
        return

    # Filter posthoc_df for this specific feature
    df_plot = posthoc_df[posthoc_df["Feature"] == feature]

    if df_plot.empty:
        return

    # Determine position of each group on the x-axis
    groups = order if order else sorted(df[x].unique())
    group_pos = {group: i for i, group in enumerate(groups)}

    # Filter and sort comparisons by group distance
    df_plot["group_pair"] = df_plot.apply(
        lambda row: tuple(sorted([row["Group 1"], row["Group 2"]])), axis=1
    )
    df_plot = df_plot.drop_duplicates(subset="group_pair")

    df_plot["x1"] = df_plot["Group 1"].map(group_pos)
    df_plot["x2"] = df_plot["Group 2"].map(group_pos)
    df_plot["distance"] = abs(df_plot["x1"] - df_plot["x2"])
    df_plot = df_plot.sort_values("distance")  # shorter distances first

    max_y = df[y].max()
    # Settings for spacing
    bar_height = 0.015 * max_y  # height of the vertical lines
    text_gap = 0 * max_y  # space between top of bar and asterisk

    for i, row in enumerate(df_plot.itertuples(index=False)):
        x1, x2 = sorted([row.x1, row.x2])
        base_y = max_y + i * (bar_height + text_gap + 3*bar_height) # base for this bar
        top_y = base_y + bar_height
        pval = row[df_plot.columns.get_loc("adj p-val")]

        # Draw line
        ax.plot([x1, x1, x2, x2], [base_y, top_y, top_y, base_y], lw=1.2, c='k')

        # Determine asterisk label
        if pval < 0.0001:
            star = '****'
        elif pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        else:
            star = 'ns'

        # Place text
        ax.text(
            (x1 + x2) / 2, top_y + text_gap,
            star, ha='center', va='baseline', color='k', fontsize=12
        )

    # Optionally annotate the test name
    if test_name:
        ax.text(
            0.5, 1.02, test_name, ha='center', va='bottom', transform=ax.transAxes, fontsize=10, style='italic'
        )

