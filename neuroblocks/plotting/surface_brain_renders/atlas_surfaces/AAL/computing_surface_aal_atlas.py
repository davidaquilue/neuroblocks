"""
This is legacy code of how we went from the raw atlas in volumetric space to the
surface atlas for plotting the renders.
"""

import os
from collections import Counter

import numpy as np
import nibabel as nib
from nilearn import datasets
from neuromaps.transforms import mni152_to_fslr

aal = datasets.fetch_atlas_aal()

# Keep only the first 90 ROIs (cortical regions)
# Remap left regions (_L, odd positions) to 1-45
# Remap right regions (_R, even positions) to 46-90
# This matches the Schaefer convention: parcelValues = [left1..left45, right1..right45]
img = nib.load(aal.maps)
data = np.array(img.dataobj)

old_indices = [int(idx) for idx in aal.indices[1:91]]  # skip Background
new_data = np.zeros_like(data)

for i in range(45):
    new_data[data == old_indices[2 * i]] = i + 1        # _L regions → 1-45
    new_data[data == old_indices[2 * i + 1]] = i + 46   # _R regions → 46-90

img_remapped = nib.Nifti1Image(new_data, img.affine, img.header)
l_gii, r_gii = mni152_to_fslr(img_remapped, fslr_density='32k', method='nearest')

# Note: some midline regions (e.g. Hippocampus, ParaHippocampal, Amygdala) may
# appear in the opposite hemisphere surface. This is handled in rendersurface_atlas.m
# by scanning both surfaces for all labels.
nib.save(l_gii, "./AAL.32k.L.label.gii")
nib.save(r_gii, "./AAL.32k.R.label.gii")
