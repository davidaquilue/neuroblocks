"""
In this script we include a function that plots a boxplot and performs statistical
testing.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from statsmodels.formula.api import ols
import statsmodels.api as sm


def boxplot_with_stats(data, group_col, value_col, method="auto", ax=None, title=""):
    """
    Plots boxplots of a continuous variable grouped by a categorical variable,
    performs statistical tests, and annotates significant differences.

    Parameters:
    - df: pandas DataFrame
    - group_col: column name (str) for group labels
    - value_col: column name (str) for continuous variable
    - method: 'auto', 'anova', or 'kruskal'. Determines the global test.
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

    ax = sns.boxplot(data=data, x=group_col, y=value_col, ax=ax, hue=group_col)
    sns.stripplot(
        data=data,
        x=group_col,
        y=value_col,
        color="black",
        size=4,
        alpha=0.6,
        jitter=True,
        ax=ax,
    )

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
