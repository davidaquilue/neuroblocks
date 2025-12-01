"""
Module containing functions to compute mean values of a given nifti/3d neuroimaging
volume in given atlas-based ROIs.

We will mostly implemented these techniques for PET SUVR, but they can be extended to
different measures.
"""
import numpy as np
import nibabel as nib

from nilearn.image import resample_to_img

def compute_region_volumes(atlas_img_path, atlas_labels):
    """
    Compute volume (mm^3) of each region in a labeled atlas.

    Parameters
    ----------
    atlas_img_path : str
        Path to the atlas NIfTI (integer labels).
    atlas_labels : list
        Region indexes in iterable form

    Returns
    -------
    list
        Volumes (volume_mm3) in list-form, following ordering in atlas_labels
    """
    # load atlas image
    img = nib.load(atlas_img_path)
    data = img.get_fdata().astype(int)

    # voxel volume in mm^3
    voxel_vol = np.abs(np.linalg.det(img.affine[:3, :3]))

    region_vols = []
    for idx in atlas_labels:
        n_vox = np.sum(data == idx)
        region_vols.append(n_vox * voxel_vol)

    return region_vols


def compute_roi_avg_value(vol_arr, atlas_arr, atlas_label, type_stat="mean"):
    """
    Compute avg value for a given ROI in a 3D neuroimaging volume

    Parameters
    ----------
    vol_arr : array
        array with volumetric data
    atlas_arr : array
        Atlas array with volumetric data, voxels are assigned a value corresponding
        to the ROI
    atlas_label : int
        Index of the region from the atlas that we will avg img values.
    type_stat : str
        Type of statistic to compute values over the voxels in the ROI (mean, median,
         sum)

    Returns
    -------
    float
        Average value in area
    """
    mask_roi = atlas_arr == atlas_label
    if type_stat == "mean":
        avg_value = np.mean(vol_arr[mask_roi])
    elif type_stat == "median":
        avg_value = np.median(vol_arr[mask_roi])
    elif type_stat == "sum":
        avg_value = np.sum(vol_arr[mask_roi])
    else:
        raise ValueError("Type statistic must be 'mean' or 'median'")
    return avg_value


def compute_metaroi_avg_value(
        img_path,
        atlas_img_path,
        atlas_labels,
        volume_weighted=False,
        type_stat_roi="mean"
):
    """
    Compute average value in a given meta-ROI from a volumetric neuroimaging image

    Parameters
    ----------
    img_path: path-like
        Path to the 3D volume/ image in NIFTI format.
    atlas_img_path : path_like
        Path to the atlas/parcellation image in NIFTI format.
    atlas_labels : list
        List of indexs of the metaroi regions from the atlas.
    volume_weighted: bool
        Whether to compute weighted average value, weighting by volume.
    type_stat_roi : str
        Type of statistic to compute value over the voxels in each ROI (mean, median,
         sum)

    Returns
    -------
    float
        Average SUVR in area
    """
    # Load the PET and atlas volumes and get array data
    img = nib.load(img_path)
    atlas_img = nib.load(atlas_img_path)
    # Resample to have the same array dimensions
    atlas_img = resample_to_img(atlas_img, img, interpolation="nearest")
    # Once resampled obtain array data
    atlas_arr = np.squeeze(atlas_img.get_fdata().astype(int))
    vol_arr = np.squeeze(img.get_fdata())
    # Compute mean/median SUVRs for each fo the SUVRs
    rois_values = []
    for atlas_label in atlas_labels:
        rois_values.append(
            compute_roi_avg_value(vol_arr, atlas_arr, atlas_label,
                                  type_stat=type_stat_roi)
        )
    # Compute average in the metaroi
    if volume_weighted:
        vols = compute_region_volumes(atlas_img_path, atlas_labels)
        metaroi_value = np.average(rois_values, weights=vols)

    else:
        metaroi_value = np.mean(rois_values)
    return metaroi_value
