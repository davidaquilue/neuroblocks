# neuroblocks

Package description
---------------------
This package accumulates functions that serve the typical Whole Brain Modelling 
project on Alzheimer's Disease research.

Folders
------------

### neuroblocks
The `src` directory contains a set of modules that help manage data loading,
path management and computation of results.

These are straightforward, will be further documented as the pipeline grows.

- `data_gathering`: 
- `data_preparation`: 
- `pipelines`:  
- `feature_extraction`:
- `plotting`:
- `atlasses`: Contains a series of atlases images for different useful parcellations that we will 
use throughout our work
 

### tests
The `studies` folder contains a set of subfolders, each of them containing 
the scripts used to compute and analyze results in different studies.

### Included atlasses

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

### MATLAB (brain renders)
The surface brain renders in `neuroblocks.plotting.plot_brain_renders` run in
MATLAB, called from Python through the MATLAB Engine API. The engine is not in
`requirements.txt` because its version must match the installed MATLAB release:
`matlabengine` 24.2.x for R2024b, 26.2.x for R2026b, and so on (see the
[version table](https://pypi.org/project/matlabengine/#history)). Install the
one matching your release, e.g. for R2026b:

```
pip install "matlabengine==26.2.*"
```

The renders also need the `gifti` function to read the surface files, which is
not bundled with neuroblocks. It comes with
[SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) (or the standalone
[GIfTI toolbox](https://github.com/gllmflndn/gifti)), whose folder must be on
the MATLAB path every time MATLAB starts, including when started by the engine.
Either add it once and save the path from MATLAB:

```matlab
addpath('C:\path\to\spm'); savepath
```

or, if `savepath` lacks write permission (common after a fresh install), add the
`addpath` line to a `startup.m` file in your MATLAB `userpath` folder (usually
`Documents\MATLAB`). Only the SPM root folder is needed, not its subfolders.
Check it with `which gifti` from MATLAB.

*Additional Technical Details such as the use of GPU or other software*
