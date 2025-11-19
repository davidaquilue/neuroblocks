"""
Here we will include functions used for BOLD Functional Connectivity.

We will keep expanding and borrow functions from other repositories...

"""

import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018


def compute_fc(bold, fisher_z=True):
    """
    Compute ROI x ROI functional connectivity from BOLD time series.

    Parameters
    ----------
    bold : ndarray, shape (n_times, n_rois)
        Preprocessed BOLD time series.
    fisher_z : bool, optional (default=True)
        If True, apply Fisher r-to-z transform to correlations.

    Returns
    -------
    fc : ndarray, shape (n_rois, n_rois)
        Functional connectivity matrix.
    """

    # Pearson correlation across ROIs (numpy does corrcoef across rows by default)
    # So we transpose to get corr across columns (ROIs)
    r = np.corrcoef(bold, rowvar=False)

    # Fisher r->z transform (optional)
    if fisher_z:
        # clip r slightly to avoid divide-by-zero or inf at r = ±1
        r = np.clip(r, -0.999999, 0.999999)
        r = np.arctanh(r)  # Fisher z-transform

    # Zero the diagonal (autocorrelations)
    np.fill_diagonal(r, 0.0)

    return r


YEO_ORDERINGS = {
    7: ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'],
    17: [
        'VisCent','VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB',
        'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB', 'ContA', 'ContB',
        'ContC', 'DefaultA', 'DefaultB', 'DefaultC'
    ]
}


def schaefer_fc_to_yeo_subnetworks_fc(schaefer_fc, n_yeo=7):
    """
    Convert a Schaefer ROI x ROI FC matrix into a Yeo-network x Yeo-network FC matrix.
    Returns both the matrix and the network ordering used.
    """

    # -------------------------------
    # 1. Load atlas info
    # -------------------------------
    n_rois = schaefer_fc.shape[0]
    atlas = fetch_atlas_schaefer_2018(n_rois, n_yeo)
    roi_labels = atlas["labels"][1:]  # skip background

    # Extract network names (second item in label, e.g., LH_Vis_3 → "Vis")
    roi_networks = np.array([lab.split("_")[2] for lab in roi_labels])

    # -------------------------------
    # 2. Pick consistent ordering
    # -------------------------------
    network_order = YEO_ORDERINGS[n_yeo]

    # Map each network name to an integer index
    net_to_index = {net: i for i, net in enumerate(network_order)}

    # Convert ROI-level network labels → integers
    roi_to_net = np.array([net_to_index[net] for net in roi_networks])

    # -------------------------------
    # 3. Build network × network FC
    # -------------------------------
    net_fc = np.zeros((n_yeo, n_yeo))

    for i in range(n_yeo):
        idx_i = np.where(roi_to_net == i)[0]

        for j in range(n_yeo):
            idx_j = np.where(roi_to_net == j)[0]

            block = schaefer_fc[np.ix_(idx_i, idx_j)]

            # remove diagonal on within-network blocks
            if i == j:
                block = block[~np.eye(len(idx_i), dtype=bool)]

            net_fc[i, j] = block.mean() if block.size > 0 else np.nan

    # -------------------------------
    # 4. Return: (FC, ordering)
    # -------------------------------
    return net_fc
