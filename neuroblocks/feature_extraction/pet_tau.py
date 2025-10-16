"""
Module containing functions for Tau PET analyses.

Two main functions currently added to the module:
- Computing Tau SUVR in PET-Tau ROIs (obtained from teh CenTauR project)
- Determining Tau positivity based on user-provided ROIs, with different
methodologies documented in [1, 2, 3]

[1] Chaggar et al. 2025 - Plos One
[2] Vogel et al. 2020 Nature Communications
[3] Ossenkoppele et al. 2022 Nature Medicine
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def fit_gmms_and_compare(values, *, random_state=0, n_init=10, create_plot=False,
                         bins=40):
    """
    Fit 1-component and 2-component Gaussian Mixture Models to 1D data and compare AIC scores.

    Parameters
    ----------
    values : array-like
        1D numeric data.
    random_state : int or None
        Random seed for GMM initialization.
    n_init : int
        Number of initializations for GaussianMixture to reduce chance of bad local minima.
    create_plot : bool
        If True, display a histogram of the data with the fitted GMM density overlays.
    bins : int
        Number of bins for histogram (only used if show_plot True).

    Returns
    -------
    result : dict
        A dictionary containing keys:
         - 'n_samples': number of samples used
         - 'aic': {1: aic_for_1comp, 2: aic_for_2comp}
         - 'bic': {1: bic_for_1comp, 2: bic_for_2comp}
         - 'log_likelihood': {1: ll per sample for 1-comp, 2: ...}  # average per-sample
         - 'models': {1: fitted GaussianMixture, 2: fitted GaussianMixture}
         - 'best_n_components': 1 or 2 (based on lower AIC; ties prefer fewer components)
         - 'plot': fig element with the plot, if create_plot is True
    Notes
    -----
    Uses sklearn.mixture.GaussianMixture.aic / bic. If data is constant or too small, the function
    handles the edge cases gracefully (returns np.inf for AIC where model cannot be fit reliably).
    """
    # Convert and sanitize input
    x = np.asarray(values, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size

    result = {
        'n_samples': n,
        'aic': {1: np.inf, 2: np.inf},
        'bic': {1: np.inf, 2: np.inf},
        'log_likelihood': {1: -np.inf, 2: -np.inf},
        'models': {1: None, 2: None},
        'best_n_components': None
    }

    if n == 0:
        raise ValueError("No valid (non-NaN) samples provided.")

    # Reshape for sklearn which expects (n_samples, n_features)
    X = x.reshape(-1, 1)

    for k in (1, 2):
        # Require at least k samples to fit k components meaningfully
        if n < k:
            continue
        # If data are all (nearly) identical, fitting >1 component is meaningless and often fails.
        if np.isclose(x.max(), x.min()) and k > 1:
            # keep model as None and leave AIC as inf
            continue
        try:
            gm = GaussianMixture(n_components=k, covariance_type='full',
                                 n_init=n_init, random_state=random_state)
            gm.fit(X)
            aic = gm.aic(X)
            bic = gm.bic(X)
            # compute average log-likelihood per sample (score returns average per-sample)
            ll_per_sample = gm.score(X)
            result['aic'][k] = aic
            result['bic'][k] = bic
            result['log_likelihood'][k] = ll_per_sample
            result['models'][k] = gm
        except Exception as e:
            # fitting failed (e.g. singular covariances). Leave defaults and include debug note.
            result['models'][k] = None
            result['aic'][k] = np.inf
            result['bic'][k] = np.inf
            result['log_likelihood'][k] = -np.inf

    # Decide best by AIC (lower is better). If tie, pick the smaller model.
    aic1 = result['aic'][1]
    aic2 = result['aic'][2]
    if np.isfinite(aic1) or np.isfinite(aic2):
        if aic1 <= aic2:
            result['best_n_components'] = 1
        else:
            result['best_n_components'] = 2
    else:
        result['best_n_components'] = None

    if create_plot:
        # Plot histogram and overlay densities
        xs = np.linspace(x.min() - (abs(x.max() - x.min()) + 1e-6) * 0.1,
                         x.max() + (abs(x.max() - x.min()) + 1e-6) * 0.1, 1000)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.hist(x, bins=bins, density=True, alpha=0.6)

        for k in (1, 2):
            gm = result['models'][k]
            if gm is None:
                continue
            # compute mixture density
            logprob = gm.score_samples(xs.reshape(-1, 1))
            pdf = np.exp(logprob)
            # overlay density (do not set explicit colors/styles here)
            ax.plot(xs, pdf, label=f'GMM {k} comp, AIC={result["aic"][k]:.1f}')
            # also plot individual components
            responsibilities = gm.predict_proba(xs.reshape(-1, 1))
            for j in range(gm.weights_.shape[0]):
                mean = gm.means_.ravel()[j]
                var = gm.covariances_.ravel()[j] if gm.covariances_.ndim == 1 else \
                gm.covariances_[j].ravel()[0]
                comp_pdf = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
                    -0.5 * (xs - mean) ** 2 / var)
                ax.plot(xs, gm.weights_[j] * comp_pdf, linestyle='--',
                         label=f'comp {j + 1} of {k}')

        ax.legend()
        ax.set_title('Data histogram and fitted GMM densities')
        ax.set(xlabel='Value', ylabel='Density')
        plt.tight_layout()
        result["plot"] = fig

    return result


def gmm_threshold_for_right_component(gm):
    """
    Compute the threshold x* where the posterior probability of belonging to the
    higher-mean component is 0.5.

    Parameters
    ----------
    gm : sklearn.mixture.GaussianMixture
        A fitted 2-component GaussianMixture (1D).

    Returns
    -------
    threshold : float
        The value x* such that P(component with higher mean | x*) = 0.5.
        Returns None if no valid threshold exists.
    """
    if gm.n_components != 2:
        raise ValueError("This function requires a 2-component GMM.")

    means = gm.means_.ravel()
    variances = (gm.covariances_.ravel() if gm.covariances_.ndim == 1
                 else gm.covariances_.reshape(-1, 1).ravel())
    weights = gm.weights_

    # Identify left and right components (by mean)
    idx_left, idx_right = np.argsort(means)
    mu1, mu2 = means[idx_left], means[idx_right]
    var1, var2 = variances[idx_left], variances[idx_right]
    w1, w2 = weights[idx_left], weights[idx_right]

    # Equation: w1 * N(x|mu1,var1) = w2 * N(x|mu2,var2)
    # Rearranged to solve for x
    a = var2 - var1
    b = 2 * (mu2*var1 - mu1*var2)
    c = var2*mu1**2 - var1*mu2**2 + 2 * var1 * var2 * np.log(
        (w2 * np.sqrt(var1)) / (w1 * np.sqrt(var2))
    )

    if np.isclose(a, 0):  # equal variances => linear equation
        if np.isclose(b, 0):
            return None  # degenerate
        x_star = -c / b
        return float(x_star)

    # quadratic: a*x^2 + b*x + c = 0
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None

    sol1 = (-b + np.sqrt(disc)) / (2 * a)
    sol2 = (-b - np.sqrt(disc)) / (2 * a)

    # pick the solution between the means
    candidates = [sol for sol in (sol1, sol2) if mu1 < sol < mu2]
    if not candidates:
        return None

    return float(candidates[0])


def cohort_tau_positivity_in_roi(
        roi_suvr_values, methodology="Chaggar", create_plot=False, cn_bool_arr=None
):
    """
    Computes Tau positivity for a given ROI based on the statistical values in a
    cohort of participants.

    If methodology == "Chaggar" [1, 2]
    - A 1 component GMM model is fit to the data
    - A 2 component GMM model is fit to the data
    - The fit of both models is compared, if 2 component fit is better, next step,
    else returns NAN positivity for all subjects.
    - Tau positivity is determined if a given value has P > 50% of belonging to the
    second (higher mean) component from the 2-component GMM model.

    If methodology == "Ossenkoppele" [3]
    - Mean healthy Tau (T_CN) value in the ROI is determined based on the CN (indexes
    with True in the cn_bool_arr).
    - Tau positivity is determined if ROI SUVR > T_CN + 2SD

    Parameters
    ----------
    roi_suvr_values : array-like
        (N_subs, ) array containing the mean suvr values for the subjects in the
        cohort.

    methodology : str
        Methodology that determines what tau positivity threshold methodology to apply

    create_plot : bool
        If methodology="Chaggar", whether to return a plot of the 1- and 2-component
        GMM fit, with fitting performances.

    cn_bool_arr : array-like
        (N_subs, ) array containing 1 if a subject in given index is CN or not (0).

    Returns
    -------
    tuple
    - [0] Tau positivity bool values for each subject in the cohort.
    - [1] Tau positivity threshold for the given ROI.
    - [2] (if methodology="Chaggar" and create_plot=True) figure with GMM fitting.
    """
    if methodology == "Chaggar":
        results_tau_gmm = fit_gmms_and_compare(roi_suvr_values, create_plot=create_plot)
        threshold = gmm_threshold_for_right_component(results_tau_gmm["models"][2])
        tau_positivity = np.array(roi_suvr_values) > threshold
        if create_plot:
            return tau_positivity, threshold, results_tau_gmm["plot"]
        else:
            return tau_positivity, threshold

    elif methodology == "Ossenkoppele":
        if cn_bool_arr is None:
            raise ValueError("cn_bool_arr must be array-like with same size as roi_suvr_values.")
        tau_roi_suvr_cn = np.array(roi_suvr_values)[np.array(cn_bool_arr).astype(bool)]
        mean_cn_tau = np.mean(tau_roi_suvr_cn)
        std_cn_tau = np.std(tau_roi_suvr_cn)
        threshold = mean_cn_tau + 2 * std_cn_tau
        tau_positivity = np.array(roi_suvr_values) >= threshold
        return tau_positivity, threshold
    else:
        raise ValueError("Provided methodology is not recognized.")
