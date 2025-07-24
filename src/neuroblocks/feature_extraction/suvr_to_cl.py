"""
Includes a function to obtain the Centiloid (CL) value for a given amyloid PET scanner.
"""

import re
import numpy as np
import nibabel as nib

from pathlib import Path
from nilearn.image import resample_to_img


def pet_suvr_to_centiloid(suvr_pet_path, suvr_to_cl_expr):
    """
    Transforms an amyloid PET suvr image to a single CL value, applying the
    transformation given by suvr_to_cl_expr, which must be of the type 'Ax + B'.

    It makes use of the CTX mask ROI for computing the average SUVR value that is
    later transformed to CL.

    :param suvr_pet_path: Path to the amyloid PET image in SUVR (normalized and
    following the same pre-processing steps as the ones used to calibrate the
    SUVR to CL expression Ax + B.
    :param suvr_to_cl_expr: str with expression "Ax + B" to transform from avg SUVR
    in the CTX ROI to CL.
    :return: float of CL value for the scan.
    """
    # Load the cortical VOI from GAAIN to account for avg SUVR in ROI
    path_ctx = Path(__file__).parent / ".." / ".." / "atlases/CL_voi_ctx_2mm.nii"
    ctx_voi_img = nib.load(path_ctx)

    # Load PET data
    suvr_pet_img = nib.load(suvr_pet_path)
    suvr_pet_arr = suvr_pet_img.get_fdata()[:]

    match = re.match(r"([-\d.]+)x\s*([+-]\s*[\d.]+)", suvr_to_cl_expr)
    a = float(match.group(1))
    b = float(match.group(2).replace(" ", ""))
    # Resample cortical target VOI to match PET image
    resampled_ctx = resample_to_img(ctx_voi_img, suvr_pet_img, interpolation="nearest")
    ctx_voi_mask = resampled_ctx.get_fdata()
    # Compute mean SUVR in the target VOI and transform to CL
    avg_suvr_ctx = np.mean(suvr_pet_arr[ctx_voi_mask == 1])
    return a * avg_suvr_ctx + b
