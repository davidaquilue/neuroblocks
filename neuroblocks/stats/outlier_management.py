"""
This module contains functions to manage (mostly remove) outliers from sets of data (
lists and dataframes)

author= "David Aquilue-Llorens"
contact = "david.aquilue@upf.edu"
date = 24/07/2025
"""

import numpy as np


def remove_outliers(list_results, percentiles=(5, 95), return_nan=True):
    """
    Remove outliers outside selected percentiles from a list. It either deletes
    them from the list if return_nan=False, or changes them as np.nan if
    return_nan=True.

    Parameters:
    - list_results (list): List of numerical values.
    - percentiles (tuple): A tuple (low, high) specifying the lower and upper
        percentiles.
    - return_nan (bool): If True, replaces outliers with np.nan. If False, removes them.

    Returns:
    - list: List with outliers either removed or replaced with np.nan.
    """
    if not list_results:
        return []

    arr = np.array(list_results, dtype=float)

    low_percentile, high_percentile = np.percentile(arr[~np.isnan(arr)], percentiles)

    if return_nan:
        result = [x if low_percentile <= x <= high_percentile else np.nan for x in arr]
    else:
        result = [x for x in arr if low_percentile <= x <= high_percentile]

    return result


def outliers_to_nans_in_df(df_data, columns=None, percentiles=(5, 95)):
    """
    Removes outliers from the columns if they are outside the given percentiles.
    If no columns are provided, all columns are analyzed and outliers in each column
    removed.
    """
    if columns is None:
        columns = df_data.columns

    for col in columns:
        df_data[col] = remove_outliers(df_data[col], percentiles, return_nan=True)

    return df_data
