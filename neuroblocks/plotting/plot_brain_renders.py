"""
Module that calls Matlab Functions to generate Brain Renders from parcellated arrays
of data.

Requires a Matlab installation.

"""
import matlab
import matlab.engine
import numpy as np
from pathlib import Path

eng = matlab.engine.start_matlab()
root_file = Path(__file__).resolve().parent
eng.addpath(str(root_file / "surface_brain_renders"))

def plot_4_views_surface_from_parcellation(
    parcellation_values,
    dir_results,
    filename_root,
    parcellation_str,
    cmap='BrBG5',
    vmin=None,
    vmax=None,
    colormap_mode=None,
    surface_type=2,
    title_text=None,
    plot_flats=False,
    save_as_img=False,
):
    """
    Function that calls Matlab Function to generate Brain Renders from parcellated
    arrays.

    Returns the figures in .pdf, to maintain vector graphics.

    Recommended cmaps:
    - Pos/Neg values around 0: BrBG5, RdBu11, PuOr11, PiYg11 (11 or 5 are similar)
    - Gradients: Purples9, GnBu8, Greens8, Oranges8
    - Viridis-like: YlGnBu9, YlOrRd9

    :param parcellation_values: vector of parcel-wise scalar values
    :param dir_results: directory where to store the figure
    :param filename_root: file_name root for the figure (default: 4views)
    :param parcellation_str: string, descriptive name of atlas (e.g. 'Schaefer400')
    :param cmap: colormap name (Matlab Based! Not classical from Python)
    :param vmin: Minimum value to display
    :param vmax: Maximum value to display
    :param colormap_mode: Colormap mode to use
    :param surface_type: 1=mid, 2=inflated, 3=very inflated
    :param title_text: string to plot as title (best avoid, not too clean)
    :param plot_flats: whether to plot flattened cortical surfaces,
        changes structure of subplots.
    :param save_as_img: whether to save the figure as an image (.png). Defaults to
        False, saving the figure as a vector graphics in .pdf.
    """
    # Ensure parcellation_values is a MATLAB double vector regardless of
    # input type (Python list, numpy array, etc.)
    if not isinstance(parcellation_values, matlab.double):
        parcellation_values = matlab.double(
            np.asarray(parcellation_values, dtype=float).tolist()
        )

    if vmin is None:
        vmin = []
    if vmax is None:
        vmax = []
    if colormap_mode is None:
        colormap_mode = []
    if title_text is None:
        title_text = []

    eng.rendersurface_atlas(
        parcellation_str,
        parcellation_values,
        str(dir_results),
        filename_root,
        vmin,
        vmax,
        colormap_mode,
        cmap,
        surface_type,
        title_text,
        plot_flats,
        save_as_img,
        nargout=0
    )

