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


def _get_schaefer_roi_to_network_map(n_rois, n_yeo=7):
    """
    Build a mapping from Schaefer ROI indices to Yeo network indices.

    Returns
    -------
    roi_to_net : ndarray of int, shape (n_rois,)
        Network index for each ROI.
    network_order : list of str
        Ordered network names corresponding to network indices.
    """
    atlas = fetch_atlas_schaefer_2018(n_rois, n_yeo)
    roi_labels = atlas["labels"][1:]  # skip background
    roi_networks = np.array([lab.split("_")[2] for lab in roi_labels])
    network_order = YEO_ORDERINGS[n_yeo]
    net_to_index = {net: i for i, net in enumerate(network_order)}
    roi_to_net = np.array([net_to_index[net] for net in roi_networks])
    return roi_to_net, network_order


def compute_network_fc_matrix(schaefer_fc, n_yeo=7):
    """
    Convert a Schaefer ROI x ROI FC matrix into a Yeo-network x Yeo-network FC matrix.
    Returns both the matrix and the network ordering used.
    """

    n_rois = schaefer_fc.shape[0]
    roi_to_net, _ = _get_schaefer_roi_to_network_map(n_rois, n_yeo)

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

    return net_fc


def compute_intra_inter_fc(schaefer_fc, networks, n_yeo=7):
    """
    Compute intra- and inter-network FC for a subset of Yeo networks.

    Parameters
    ----------
    schaefer_fc : ndarray, shape (n_rois, n_rois)
        Schaefer ROI-level FC matrix (e.g., Fisher z-transformed).
    networks : list of str
        Subset of Yeo network names to compute metrics for
        (e.g., ["Default", "Cont", "SalVentAttn"]).
    n_yeo : int, optional (default=7)
        Number of Yeo networks (7 or 17).

    Returns
    -------
    result : dict
        {network_name: {"intra": float, "inter": float}} for each
        requested network. "intra" is the mean within-network FC (excluding
        diagonal), "inter" is the mean FC between the network's parcels and
        all parcels belonging to other networks.
    """
    n_rois = schaefer_fc.shape[0]
    roi_to_net, network_order = _get_schaefer_roi_to_network_map(n_rois, n_yeo)
    net_to_index = {net: i for i, net in enumerate(network_order)}

    result = {}
    for net_name in networks:
        net_idx = net_to_index[net_name]
        in_net = np.where(roi_to_net == net_idx)[0]
        out_net = np.where(roi_to_net != net_idx)[0]

        # Intra-network: within-network block, excluding diagonal
        intra_block = schaefer_fc[np.ix_(in_net, in_net)]
        intra_vals = intra_block[~np.eye(len(in_net), dtype=bool)]
        intra_fc = intra_vals.mean() if intra_vals.size > 0 else np.nan

        # Inter-network: between this network's parcels and all other parcels
        inter_block = schaefer_fc[np.ix_(in_net, out_net)]
        inter_fc = inter_block.mean() if inter_block.size > 0 else np.nan

        result[net_name] = {"intra": intra_fc, "inter": inter_fc}

    return result


def compute_pairwise_network_fc(schaefer_fc, networks, n_yeo=7):
    """
    Compute pairwise between-network FC for a subset of Yeo networks.

    Uses compute_network_fc_matrix to get the full (n_yeo, n_yeo) matrix,
    then extracts the off-diagonal entries for the requested network pairs.

    Parameters
    ----------
    schaefer_fc : ndarray, shape (n_rois, n_rois)
        Schaefer ROI-level FC matrix (e.g., Fisher z-transformed).
    networks : list of str
        Subset of Yeo network names (e.g., ["Default", "DorsAttn", "SalVentAttn"]).
    n_yeo : int, optional (default=7)
        Number of Yeo networks (7 or 17).

    Returns
    -------
    result : dict
        {(net_a, net_b): float} for each unique pair from the requested networks.
        The value is the mean FC between all parcels in net_a and all parcels in net_b.
    """
    yeo_fc = compute_network_fc_matrix(schaefer_fc, n_yeo)
    network_order = YEO_ORDERINGS[n_yeo]
    net_to_index = {net: i for i, net in enumerate(network_order)}

    result = {}
    for a_i, net_a in enumerate(networks):
        for net_b in networks[a_i + 1:]:
            idx_a = net_to_index[net_a]
            idx_b = net_to_index[net_b]
            result[(net_a, net_b)] = yeo_fc[idx_a, idx_b]

    return result


def compute_gbc(fc):
    """
    Compute Global Brain Connectivity (GBC) for each ROI.

    GBC is the row-wise mean of the FC matrix, excluding the diagonal.

    Parameters
    ----------
    fc : ndarray, shape (n_rois, n_rois)
        Functional connectivity matrix (diagonal should be 0).

    Returns
    -------
    gbc : ndarray, shape (n_rois,)
        Global brain connectivity per ROI.
    """
    n = fc.shape[0]
    # Sum each row excluding diagonal, divide by (n-1)
    row_sums = fc.sum(axis=1)  # diagonal is already 0 from compute_fc
    return row_sums / (n - 1)


def compute_bold_sd(bold_ts):
    """
    Compute temporal standard deviation of BOLD signal per region.

    Parameters
    ----------
    bold_ts : ndarray, shape (n_times, n_rois)
        Preprocessed BOLD time series.

    Returns
    -------
    sd : ndarray, shape (n_rois,)
        Temporal standard deviation per ROI.
    """
    return np.std(bold_ts, axis=0, ddof=1)
