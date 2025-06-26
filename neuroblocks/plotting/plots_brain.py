"""
Functions to plot data in brain
"""
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_stat_map
from ..data_preparation.parcellation_tools import get_parcellation_nifti


def parcellation_to_brain_heatmap(data_parcellated, parcellation_type, output_path,
                                  **kwargs_stat_map):
    """
    A parcellation is passed, and a 3D volume is generated which is then used to plot a
    3-view plot of heatmap.

    Stores the image in the output_path
    :param data_parcellated: Data in parcellated form (n_rois, )
    :param parcellation_type: string for the type of parcellation (Glasser,
    SchaeferN, DBS80)
    :param output_path: path where to store the figure
    :param kwargs_stat_map: arguments to be passed to the
    nilearn.plotting.plot_stat_map function for the plot.
    :return:
    """
    parcellation_nifti = get_parcellation_nifti(parcellation_type)
    parcellation_data = parcellation_nifti.get_fdata()

    # Create an empty 3D array for the new NIfTI image
    data_nifti = np.zeros(parcellation_data.shape)

    # Loop through region labels and assign the scalar value to each voxel in the region
    for region_label in range(1, len(data_parcellated) + 1):
        mask = (parcellation_data == region_label)
        data_nifti[mask] += data_parcellated[region_label - 1]  # Correct scalar assignment

    # Create a new NIfTI image
    data_img = nib.Nifti1Image(data_nifti, parcellation_nifti.affine)

    # Display the map on the MNI152 template
    plot_stat_map(data_img, output_file=output_path, **kwargs_stat_map)
