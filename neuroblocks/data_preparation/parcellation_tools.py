"""
This script contains functions related to parcellations:

- get_parcellater: returns the parcellator object for any of the implemented
parcellations
- get_parcellation_nifti: returns a nifti Image of the atlas for the implemented
parcellation.
- parcellation_to_volume: Transforms parcellated_data in (n_rois, ) to a volumetric
image in the atlas' space.

Current implemented parcellations:
- SchaeferN (N=100, 200, 400, 1000)
- Glasser
- DBS80
"""
import re
import numpy as np
import nibabel as nib
import nilearn.datasets
from pathlib import Path

from neuromaps.parcellate import Parcellater


def get_parcellater(parcellation_str):
    """
    Function that returns the parcellator object for any of the implemented
    parcellations. Implemented parcellations currently are:
    - SchaeferN (N=100, 200, 400, 1000)
    - Glasser
    - DBS80

    :param parcellation_str: str containing the code for the implemented parcellation.
    :return: Returns the fit parcellater object to apply a parcellation, as well as
    the file extension that is attached to the end of the parcellated file.
    """
    if "Schaefer" in parcellation_str:
        nrois = int(re.findall(r"\d+", parcellation_str)[0])
        # First we fetch the parcellation atlas
        schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=nrois, resolution_mm=2)
        # Initialize the Parcellater
        parcellater = Parcellater(schaefer["maps"], "MNI152")
        file_ext = f"Schaefer{nrois}"

    elif "Glasser" in parcellation_str:
        glasser_path = (Path(__file__).parent / ".." / ".." / "atlases" / "Glasser" /
                        "glasser_MNI152NLin6Asym_labels_p20.nii.gz").resolve()
        parcellater = Parcellater(glasser_path.resolve(), "MNI152")
        file_ext = "Glasser"

    elif "DBS80" in parcellation_str:
        dbs_path = (Path(__file__).parent / ".." / ".." / "atlases" / "DBS80" /
                    "dbs80symm_2mm.nii.gz").resolve()
        parcellater = Parcellater(dbs_path, "MNI152")
        file_ext = "DBS80"
    else:
        raise ValueError("Parcellation type is not integrated in get_parcellater")

    parcellater.fit()
    return parcellater, file_ext


def get_parcellation_nifti(parcellation_str):
    """
    Function that returns the nifti image containing the atlas of implemented
    parcellations. The implemented parcellations currently are:
    - SchaeferN (N=100, 200, 400, 1000)
    - Glasser
    - DBS80

    :param parcellation_str: str containing the code for the implemented parcellation.

    :return: Returns a nibabel.Nifti1Image with the atlas of the corresponding
    parcellation.
    """
    # Load the parcellation map
    if "Schaefer" in parcellation_str:
        nrois = int(re.findall(r"\d+", parcellation_str)[0])
        # First we fetch the parcellation atlas
        schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=nrois, resolution_mm=2)
        parcellation_nifti = nib.load(schaefer['maps'])

    elif "Glasser" in parcellation_str:
        glasser_path = (Path(__file__).parent / ".." / ".." / "atlases" / "Glasser" /
                        "glasser_MNI152NLin6Asym_labels_p20.nii.gz").resolve()
        parcellation_nifti = nib.load(glasser_path)

    elif "DBS80" in parcellation_str:
        dbs_path = (Path(__file__).parent / ".." / ".." / "atlases" / "DBS80" /
                    "dbs80symm_2mm.nii.gz").resolve()
        parcellation_nifti = nib.load(dbs_path)
    else:
        raise ValueError("Parcellation is not integrated in get_parcellation_nifti")

    return parcellation_nifti


def parcellation_to_volume(parcellated_data, parcellation_str):
    """
    Transformation from parcellated data in the shape (n_rois, ) to a volumetric
    image in the atlas_space, using the atlas in atlas_path (atlas being the
    volumetric image where each voxel has the index of a ROI).

    :param parcellated_data: array-like of (n_rois, ) with the shape
    :param parcellation_str: string identifying the parcellation. Currently
    implemented: SchaeferN (N=100, 200, 400, 1000), Glasser, DBS80

    :return: nibabel.Nifti1Image with parcellated data into volumetric format,
    in the same space as the atlas.
    """
    atlas_nifti = get_parcellation_nifti(parcellation_str)
    atlas_data = atlas_nifti.get_fdata()
    volume_nifti = np.zeros(atlas_data.shape)
    for region_label in range(1, len(parcellated_data) + 1):
        mask = (atlas_data == region_label)
        volume_nifti[mask] += parcellated_data[region_label - 1]
    volume_nifti = nib.Nifti1Image(volume_nifti, atlas_nifti.affine)
    return volume_nifti
