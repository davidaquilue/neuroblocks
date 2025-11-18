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
from sklearn.utils import shuffle
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    formula = f"{feature} ~ C({group_col})" + "".join(
        [f" + {cov}" for cov in final_covariates]
    )
    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    p_value = anova_table.loc["C(group)", "PR(>F)"]

    # Linearity check: Pearson correlation for all numeric covariates
    linearity_corrs = {
        cov: df[[cov, feature]].corr().iloc[0, 1] for cov in final_covariates
    }

    # Homogeneity of regression slopes: test group * covariate interaction
    interaction_terms = " + ".join(
        [f"C({group_col}):{cov}" for cov in final_covariates]
    )
    formula_interaction = (
        f"{feature} ~ C({group_col}) + "
        + " + ".join([f"{cov}" for cov in final_covariates])
        + " + "
        + interaction_terms
    )
    model_interaction = smf.ols(formula_interaction, data=df).fit()
    anova_interaction = anova_lm(model_interaction, typ=2)

    slope_pvals = {
        f"{group_col}:{cov}": anova_interaction.to_dict()
        .get("PR(>F)", {})
        .get(f"C({group_col}):{cov}", None)
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


def permutation_ancova(
        df,
        feature,
        effect_term,
        covariates,
        n_perm=10000,
        random_state=0,
        group_key="group",
        verbose=True
):
    """
    Robust permutation ANCOVA using the Freedman–Lane method.
    Returns parametric F, permutation p-value, assumption checks, and effect sizes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    feature : str
        Dependent variable.
    effect_term : str
        Term to test (e.g., 'C(group)' or 'C(group):C(dataset)').
    covariates : list
        Covariate column names.
    n_perm : int
        Number of permutations.
    random_state : int
        Seed.
    group_key : str
        Group column name.
    verbose : bool
        Print assumption checks.

    Returns
    -------
    dict
        {
            "F_obs": float,
            "p_parametric": float,
            "p_permutation": float,
            "F_null": np.ndarray,
            "assumptions": dict,
            "partial_eta2": float,
            "cohen_d_residual": float
        }

    References
    ----------
    Freedman & Lane (1983). A nonstochastic interpretation of reported significance levels.
    Winkler et al. (2014). Permutation inference for the general linear model. NeuroImage.
    """
    df = df.copy()

    # Build formula strings
    cov_terms = []
    for c in covariates:
        if df[c].dtype.name == 'category' or df[c].dtype == object:
            cov_terms.append(f"C({c})")
        else:
            cov_terms.append(c)
    cov_formula = " + ".join(cov_terms)

    # FULL MODEL includes effect term
    formula_full = f"{feature} ~ {effect_term} + {cov_formula}"

    # REDUCED MODEL excludes effect term (for Freedman–Lane)
    formula_reduced = f"{feature} ~ {cov_formula}"

    # ----------- FIT FULL MODEL -----------
    # Change ancova type if we have interaction (e.g. C(group):C(dataset))
    ancova_type = 3 if len(effect_term.split(":")) > 1 else 2
    model_full = smf.ols(formula_full, data=df).fit(cov_type="HC3")
    anova_full = anova_lm(model_full, typ=ancova_type)

    # Parametric F and p-value
    F_obs = anova_full.loc[effect_term, "F"]
    p_param = anova_full.loc[effect_term, "PR(>F)"]

    # Partial eta squared
    ss_effect = anova_full.loc[effect_term, "sum_sq"]
    ss_resid = anova_full.loc["Residual", "sum_sq"]
    partial_eta2 = ss_effect / (ss_effect + ss_resid)

    # Cohen's d on residuals (residualize DV wrt covariates)
    model_reduced = smf.ols(formula_reduced, data=df).fit()
    resid_adj = df[feature] - model_reduced.fittedvalues
    # Extract groups for effect_term
    groups = df[group_key]
    group_vals = [resid_adj[groups == g].values for g in np.unique(groups)]
    if len(group_vals) == 2:
        # pooled SD
        sd_pooled = np.sqrt(
            (np.var(group_vals[0], ddof=1) + np.var(group_vals[1], ddof=1)) / 2)
        cohen_d_resid = (np.mean(group_vals[1]) - np.mean(group_vals[0])) / sd_pooled
    else:
        cohen_d_resid = np.nan  # cannot compute d for >2 levels

    # ------------------ Assumption checks ------------------
    assumptions = {}

    # Normality
    shapiro_p = shapiro(model_full.resid)[1]
    assumptions["Normality (Shapiro)"] = shapiro_p

    # Homoscedasticity
    bp_p = het_breuschpagan(model_full.resid, model_full.model.exog)[1]
    assumptions["Homoscedasticity (Breusch_Pagan)"] = bp_p

    # Linearity (Pearson correlation with numeric covariates)
    lin_dict = {}
    for c in covariates:
        if np.issubdtype(df[c].dtype, np.number):
            lin_dict[c] = df[[c, feature]].corr().iloc[0, 1]
    assumptions["Linearity (r with DV)"] = lin_dict

    # VIF for multicollinearity
    vif_df = pd.DataFrame()
    vif_df["term"] = model_full.model.exog_names
    vif_df["VIF"] = [
        variance_inflation_factor(model_full.model.exog, i)
        for i in range(model_full.model.exog.shape[1])
    ]
    assumptions["Multicollinearity (VIF)"] = vif_df

    # Group × covariate interactions to check slope homogeneity
    slope_dict = {}
    for c in covariates:
        test_formula = f"{feature} ~ {effect_term}*{c} + {' + '.join([cov for cov in cov_terms if cov != c])}"
        model_slope = smf.ols(test_formula, data=df).fit()
        anova_slope = anova_lm(model_slope, typ=ancova_type)
        term = f"{effect_term}:{c}"
        if term in anova_slope.index:
            slope_dict[term] = anova_slope.loc[term, "PR(>F)"]
    assumptions["Homogeneity of Slopes (interaction p_values)"] = slope_dict

    # Print summary
    if verbose:
        print("\n--- ANCOVA Assumption Check ---")
        for k, v in assumptions.items():
            print(f"{k}: {v}")

    # ------------------ Freedman-Lane Permutation ------------------
    resid_reduced = model_reduced.resid.values
    fitted_reduced = model_reduced.fittedvalues.values
    F_null = np.zeros(n_perm)
    # Set random seed once
    np.random.seed(random_state)

    for i in range(n_perm):
        # Let shuffle use its own random state (None = truly random each iteration)
        perm_resid = shuffle(resid_reduced, random_state=None)

        # Create permuted y values
        y_perm = fitted_reduced + perm_resid

        # Fit model on permuted data
        df_perm = df.copy()
        df_perm["_y_perm"] = y_perm

        model_perm = smf.ols(f"_y_perm ~ {effect_term} + {cov_formula}", data=df_perm).fit()
        anova_perm = anova_lm(model_perm, typ=ancova_type)

        F_null[i] = anova_perm.loc[effect_term, "F"]

    p_perm = (np.sum(F_null >= F_obs) + 1) / (n_perm + 1)

    return {
        "F_obs": F_obs,
        "p_parametric": p_param,
        "p_permutation": p_perm,
        "F_null": F_null,
        "assumptions": assumptions,
        "partial_eta2": partial_eta2,
        "cohen_d_residual": cohen_d_resid
    }


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
                p_value, test_name, assumption_summary = (
                    np.nan,
                    "ANCOVA (error)",
                    str(e),
                )

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
        eff_size_metric = get_effect_size_metric(test_name)
        # We add the groups as well
        results.append(
            [
                feature,
                test_name,
                p_value,
                eff_size_metric,
                effect_size,
                assumption_summary,
            ]
            + groups
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

    group_cols = [f"Group {i + 1}" for i in range(num_groups)]
    results_df = pd.DataFrame(
        results,
        columns=[
            "Feature",
            "Test",
            "p-val",
            "Effect Size Metric",
            "Effect Size",
            "Assumptions",
        ]
        + group_cols,
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
        group1, group2 = data_groups
        n1, n2 = len(group1), len(group2)

        # U statistic (SciPy gives U for group1 vs group2)
        u, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")

        # Get the smaller of U and its complement
        u2 = n1 * n2 - u
        u_min = min(u, u2)

        # Compute absolute rank-biserial effect size
        r_abs = 1 - (2 * u_min) / (n1 * n2)

        # Assign sign to match Cohen’s d direction (mean difference)
        r = np.sign(np.mean(group1) - np.mean(group2)) * abs(r_abs)
        return r

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


def get_effect_size_metric(test_name):
    dict_eff_size_metrics = {
        "t-test": "d",
        "Mann-Whitney U": "r",
        "ANOVA": "eta_squared",
        "ANCOVA": "eta_squared",
        "Kruskal-Wallis": "epsilon_squared",
    }
    return dict_eff_size_metrics[test_name]
