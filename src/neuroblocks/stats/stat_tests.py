"""
This module contains functions to apply statistical tests to data to assess whether
there are significant differences between groups, as well as to compute effect sizes
and additional statistical measures.

author= "David Aquilue-Llorens"
contact = "david.aquilue@upf.edu"
date = 24/07/2025
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf


def stat_tests_features(df_features, features="all", groups="all", covariates=None):
    """
    Perform statistical tests on EEG features across groups, with optional covariate correction.

    Parameters:
    - df_features (pd.DataFrame): Must include 'sub', 'class', features, and optionally covariates.
    - features (list of str or "all"): Features to test.
    - groups (list of str or "all"): Group labels to compare from 'class' column.
    - covariates (list of str or None): List of covariate column names to control for.

    Returns:
    - results_df: DataFrame with test results and adjusted p-values.
    - posthoc_df (optional): If >2 groups, post-hoc test results.
    """
    if features == "all":
        features = df_features.columns.difference(
            ["sub", "class"] + (covariates if covariates else [])
        ).tolist()

    if groups == "all":
        groups = df_features["class"].unique().tolist()

    df_features = df_features[df_features["class"].isin(groups)].copy()
    num_groups = len(groups)

    results = []
    posthoc_results = []

    for feature in features:
        if covariates:  # Linear model with covariates
            formula = f"{feature} ~ C(class)" + "".join(
                [f" + {cov}" for cov in covariates]
            )
            model = smf.ols(formula, data=df_features).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_value = anova_table.loc["C(class)", "PR(>F)"]
            test_name = "ANCOVA"
        else:  # Non-parametric or parametric test based on normality
            data_groups = [
                df_features[df_features["class"] == g][feature].dropna() for g in groups
            ]
            normality_pvals = [
                stats.shapiro(gr).pvalue for gr in data_groups if len(gr) >= 4
            ]
            normal_data = all(p > 0.05 for p in normality_pvals)

            if num_groups == 2:
                if normal_data:
                    test_name = "t-test"
                    stat, p_value = stats.ttest_ind(*data_groups, equal_var=False)
                else:
                    test_name = "Mann-Whitney U"
                    stat, p_value = stats.mannwhitneyu(
                        *data_groups, alternative="two-sided"
                    )
            else:
                if normal_data:
                    test_name = "ANOVA"
                    stat, p_value = stats.f_oneway(*data_groups)
                else:
                    test_name = "Kruskal-Wallis"
                    stat, p_value = stats.kruskal(*data_groups)
        effect_size = compute_effect_size(
            feature, df_features, groups, test_name, covariates
        )
        results.append([feature, test_name, p_value, effect_size])

        # Post-hoc if needed
        if num_groups > 2 and not covariates and p_value < 0.05:
            if test_name == "ANOVA":
                posthoc = pairwise_tukeyhsd(df_features[feature], df_features["class"])
                for i in range(len(posthoc.groupsunique)):
                    for j in range(i + 1, len(posthoc.groupsunique)):
                        group1 = posthoc.groupsunique[i]
                        group2 = posthoc.groupsunique[j]
                        idx = (
                            i * len(posthoc.groupsunique) + j - ((i + 1) * (i + 2)) // 2
                        )
                        p_adj = posthoc.pvalues[idx]
                        posthoc_results.append(
                            [feature, "Tukey HSD", group1, group2, p_adj]
                        )
            elif test_name == "Kruskal-Wallis":
                posthoc = sp.posthoc_dunn(
                    df_features, val_col=feature, group_col="class", p_adjust="fdr_bh"
                )
                for g1 in posthoc.index:
                    for g2 in posthoc.columns:
                        if g1 != g2:
                            posthoc_results.append(
                                [feature, "Dunn's Test", g1, g2, posthoc.loc[g1, g2]]
                            )

    results_df = pd.DataFrame(
        results, columns=["Feature", "Test", "p-val", "Effect Size"]
    )
    results_df["adj p-val"] = multipletests(results_df["p-val"], method="fdr_bh")[1]

    posthoc_df = pd.DataFrame(
        posthoc_results,
        columns=["Feature", "Post-hoc Test", "Group 1", "Group 2", "adj p-val"],
    )
    posthoc_df = posthoc_df[posthoc_df["adj p-val"] < 0.05]

    if not posthoc_df.empty:
        return results_df, posthoc_df
    else:
        return results_df


def compute_effect_size(
    feature, df_data, groups, test_name, covariates=None, group_key="class"
):
    """
    Computes different effect size metrics based on the statistical test applied.

    Parameters:
        feature: Column name of the feature that will be used to compute effect size
            in df.
        df_data: pd.Dataframe with all accumulated data
        groups: list of str with group names in the data. p.eg. ["HC", "AD"]
        test_name: Statistical test that has been used to compute p-value, so that
        effect size is adapted to the test.
        covariates: Whether we want to use some of the columns in df as covariates.
        group_key: Key for the "groups" column, defaults to "class".
    """

    # We rename the group key to class so that we can apply it independently of key
    df_data = df_data.rename(columns={group_key: "class"})
    data_groups = [df_data[df_data["class"] == g][feature].dropna() for g in groups]

    if test_name == "t-test":
        group1, group2 = data_groups
        n1, n2 = len(group1), len(group2)
        pooled_sd = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
            / (n1 + n2 - 2)
        )
        d = (np.mean(group1) - np.mean(group2)) / pooled_sd
        return d

    elif test_name == "Mann-Whitney U":
        u, _ = stats.mannwhitneyu(*data_groups, alternative="two-sided")
        n1, n2 = len(data_groups[0]), len(data_groups[1])
        r = 1 - (2 * u) / (n1 * n2)
        return r  # Rank-biserial correlation

    elif test_name in ["ANOVA", "ANCOVA"]:
        if covariates:
            formula = f"{feature} ~ C(class)" + "".join(
                [f" + {cov}" for cov in covariates]
            )
        else:
            formula = f"{feature} ~ C(class)"
        model = smf.ols(formula, data=df_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        ss_between = anova_table.loc["C(class)", "sum_sq"]
        ss_total = anova_table["sum_sq"].sum()
        eta_squared = ss_between / ss_total
        return eta_squared

    elif test_name == "Kruskal-Wallis":
        h_stat, _ = stats.kruskal(*data_groups)
        n = sum(len(g) for g in data_groups)
        epsilon_squared = (h_stat - len(groups) + 1) / (n - 1)
        return epsilon_squared

    return np.nan
