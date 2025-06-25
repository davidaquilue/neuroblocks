# NeuroBlocks

Package description
---------------------
This package accumulates functions that serve the typical Whole Brain Modelling 
project on Alzheimer's Disease research.

Folders
------------

### NeuroBlocks
The `src` directory contains a set of modules that help manage data loading,
path management and computation of results.

These are straightforward, will be further documented as the pipeline grows.

- `data_gathering`: 
- `data_preparation`: 
- `pipelines`:  
- `feature_extraction`:
- `plotting`:
 

### tests
The `studies` folder contains a set of subfolders, each of them containing 
the scripts used to compute and analyze results in different studies.

### atlasses
Contains a series of atlases images for different useful parcellations that we will 
use throughout our work

Currently implemented parcellations are the following:
- SchaeferN (N=100, 200, 400, 1000)
- Glasser
- DBS80

Additionally, it includes other types of masks that may be useful for further work 
such as Cortical ROI masks for Centiloid value computation or cerebellar masks for 
PET SUVR computation.


Documentation
--------------
Attach additional information


Technical details
---------------------------
The code is developed using Python `3.9`.

A miniconda virtual environment is used, containing the packages listed in the 
`requirements.txt` file.

*Additional Technical Details such as the use of GPU or other software*
