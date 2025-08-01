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
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import shapiro
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan


def ancova_with_assumption_checks(df, feature, group_col, covariates):
    df = df.copy()

    # Binarize non-numeric covariates
    binarized_covariates = []
    final_covariates = []

    for cov in covariates:
        if not np.issubdtype(df[cov].dtype, np.number):
            dummies = pd.get_dummies(df[cov], prefix=cov, drop_first=True)
            df = pd.concat([df.drop(columns=[cov]), dummies], axis=1)
            binarized_covariates.extend(dummies.columns.tolist())
        else:
            final_covariates.append(cov)

    final_covariates += binarized_covariates

    # Fit ANCOVA model
    formula = f"{feature} ~ C({group_col})" + "".join([f" + {cov}" for cov in final_covariates])
    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    p_value = anova_table.loc["C(group)", "PR(>F)"]

    # Linearity check: Pearson correlation for all numeric covariates
    linearity_corrs = {
        cov: df[[cov, feature]].corr().iloc[0, 1]
        for cov in final_covariates
    }

    # Homogeneity of regression slopes: test group * covariate interaction
    interaction_terms = " + ".join([f"C({group_col}):{cov}" for cov in final_covariates])
    formula_interaction = f"{feature} ~ C({group_col}) + " + " + ".join([f"{cov}" for cov in final_covariates]) + " + " + interaction_terms
    model_interaction = smf.ols(formula_interaction, data=df).fit()
    anova_interaction = anova_lm(model_interaction, typ=2)

    slope_pvals = {
        f"{group_col}:{cov}": anova_interaction.to_dict().get("PR(>F)", {}).get(
            f"C({group_col}):{cov}", None)
        for cov in final_covariates
    }

    # Normality of residuals
    shapiro_p = shapiro(model.resid)[1]

    # Homoscedasticity (Breusch-Pagan test)
    exog = model.model.exog
    bp_test = het_breuschpagan(model.resid, exog)
    bp_pval = bp_test[1]

    # Combine assumption info
    assumption_summary = (
        f"Linearity: {linearity_corrs} | "
        f"Slope homogeneity p: {slope_pvals} | "
        f"Shapiro-Wilk p: {shapiro_p:.3f} | "
        f"Breusch-Pagan p: {bp_pval:.3f}"
    )

    return p_value, "ANCOVA", assumption_summary


def stat_tests_features(
        df_features, features="all", groups="all", covariates=None, group_key="group"
):
    """
    Perform statistical tests on EEG features across groups, with optional covariate correction.

    Parameters:
    - df_features (pd.DataFrame): Must include 'sub', group_key, features,
        and optionally covariates.
    - features (list of str or "all"): Features to test.
    - groups (list of str or "all"): Group labels to compare from 'group' column.
    - covariates (list of str or None): List of covariate column names to control for.
    - group_key (str or None): Column name to use for groups/class

    Returns:
    - results_df: DataFrame with test results and adjusted p-values.
    - posthoc_df (optional): If >2 groups, post-hoc test results.
    """
    df_features = df_features.rename(columns={group_key: "group"})
    df_features = df_features.sort_values(by="group")
    if features == "all":
        features = df_features.columns.difference(
            ["sub", "group"] + (covariates if covariates else [])
        ).tolist()

    if groups == "all":
        groups = df_features["group"].unique().tolist()

    df_features = df_features[df_features["group"].isin(groups)].copy()
    num_groups = len(groups)

    results = []
    posthoc_results = []

    assumption_summary = None
    for feature in features:
        if covariates:
            try:
                p_value, test_name, assumption_summary = ancova_with_assumption_checks(
                    df_features, feature, group_col="group", covariates=covariates
                )
            except Exception as e:
                p_value, test_name, assumption_summary = np.nan, "ANCOVA (error)", str(e)

        else:  # Non-parametric or parametric test based on normality
            data_groups = [
                df_features[df_features["group"] == g][feature].dropna() for g in groups
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
        # We add the groups as well
        results.append(
            [feature, test_name, p_value, effect_size, assumption_summary] + groups
        )

        # Post-hoc if needed
        if num_groups > 2 and not covariates and p_value < 0.05:
            if test_name == "ANOVA":
                posthoc = pairwise_tukeyhsd(df_features[feature], df_features["group"])
                seen_pairs = set()
                for i in range(len(posthoc.groupsunique)):
                    for j in range(i + 1, len(posthoc.groupsunique)):
                        group1 = posthoc.groupsunique[i]
                        group2 = posthoc.groupsunique[j]
                        pair = tuple(sorted((group1, group2)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        idx = (
                            i * len(posthoc.groupsunique) + j - ((i + 1) * (i + 2)) // 2
                        )
                        p_adj = posthoc.pvalues[idx]
                        posthoc_results.append(
                            [feature, "Tukey HSD", group1, group2, p_adj]
                        )
            elif test_name == "Kruskal-Wallis":
                posthoc = sp.posthoc_dunn(
                    df_features, val_col=feature, group_col="group", p_adjust="fdr_bh"
                )
                seen_pairs = set()
                for g1 in posthoc.index:
                    for g2 in posthoc.columns:
                        if g1 == g2:
                            continue
                        pair = tuple(sorted((g1, g2)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        posthoc_results.append(
                            [feature, "Dunn's Test", g1, g2, posthoc.loc[g1, g2]]
                        )

    group_cols = [f"Group {i+1}" for i in range(num_groups)]
    results_df = pd.DataFrame(
        results,
        columns=["Feature", "Test", "p-val", "Effect Size", "Assumptions"] + group_cols
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
    feature, df_data, groups, test_name, covariates=None, group_key="group"
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
        group_key: Key for the "groups" column, defaults to "group".
    """

    # We rename the group key to group so that we can apply it independently of key
    df_data = df_data.rename(columns={group_key: "group"})
    data_groups = [df_data[df_data["group"] == g][feature].dropna() for g in groups]

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
            formula = f"{feature} ~ C(group)" + "".join(
                [f" + {cov}" for cov in covariates]
            )
        else:
            formula = f"{feature} ~ C(group)"
        model = smf.ols(formula, data=df_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        ss_between = anova_table.loc["C(group)", "sum_sq"]
        ss_total = anova_table["sum_sq"].sum()
        eta_squared = ss_between / ss_total
        return eta_squared

    elif test_name == "Kruskal-Wallis":
        h_stat, _ = stats.kruskal(*data_groups)
        n = sum(len(g) for g in data_groups)
        epsilon_squared = (h_stat - len(groups) + 1) / (n - 1)
        return epsilon_squared

    return np.nan
