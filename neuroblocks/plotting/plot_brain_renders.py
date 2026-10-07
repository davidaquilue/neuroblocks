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
    parcel_alpha=None,
    parcel_outline=None,
    outline_color=(0.0, 0.0, 0.0),
    outline_width=1.0,
    background_color=(0.85, 0.85, 0.85),
):
    """
    Function that calls Matlab Function to generate Brain Renders from parcellated
    arrays.

    Returns the figures in .pdf, to maintain vector graphics.

    Recommended cmaps:
    - Pos/Neg values around 0: BrBG5, RdBu11, PuOr11, PiYg11 (11 or 5 are similar)
    - Gradients: Purples9, GnBu8, Greens8, Oranges8
    - Viridis-like: YlGnBu9, YlOrRd9

    Transparent thresholding (Taylor et al. 2023, NeuroImage 274:120138) can be
    drawn with ``parcel_alpha`` and ``parcel_outline``: e.g. supra-threshold
    parcels with alpha 1 and an outline, sub-threshold parcels faded toward
    ``background_color``. The renderer does not compute any statistics; the
    caller decides alpha and outlines. The medial wall is always drawn in
    [0.95, 0.95, 0.95].

    :param parcellation_values: vector of parcel-wise scalar values. NaN values
        are drawn in ``background_color`` (alpha 0).
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
        False, saving the figure as a vector graphics in .pdf (brain panels are
        embedded as images, colorbar and title are vector; with outlines the
        panels are embedded at 600 dpi).
    :param parcel_alpha: array-like of floats in [0, 1], one per parcel. 1 draws
        the parcel at full color, 0 draws it in ``background_color``, values in
        between blend the two. None (default) means all ones.
    :param parcel_outline: array-like of bools, one per parcel. True draws a closed
        contour around the parcel (adjacent outlined parcels share one contour).
        None (default) draws no outlines.
    :param outline_color: RGB color of the outlines, values in [0, 1].
    :param outline_width: line width of the outlines.
    :param background_color: RGB color, values in [0, 1], that faded parcels
        blend into.
    :raises ValueError: if ``parcel_alpha`` or ``parcel_outline`` do not have one
        element per parcel, or if any alpha is outside [0, 1].
    """
    # Ensure parcellation_values is a MATLAB double vector regardless of
    # input type (Python list, numpy array, etc.)
    if not isinstance(parcellation_values, matlab.double):
        parcellation_values = matlab.double(
            np.asarray(parcellation_values, dtype=float).ravel().tolist()
        )
    n_parcels = int(np.prod(parcellation_values.size))

    if parcel_alpha is None:
        parcel_alpha = []
    else:
        parcel_alpha = np.asarray(parcel_alpha, dtype=float).ravel()
        if parcel_alpha.size != n_parcels:
            raise ValueError(
                f"parcel_alpha has {parcel_alpha.size} elements, expected "
                f"{n_parcels} (one per value in parcellation_values)"
            )
        if not np.all((parcel_alpha >= 0) & (parcel_alpha <= 1)):
            raise ValueError("parcel_alpha values must be within [0, 1]")
        parcel_alpha = matlab.double(parcel_alpha.tolist())

    if parcel_outline is None:
        parcel_outline = []
    else:
        parcel_outline = np.asarray(parcel_outline, dtype=bool).ravel()
        if parcel_outline.size != n_parcels:
            raise ValueError(
                f"parcel_outline has {parcel_outline.size} elements, expected "
                f"{n_parcels} (one per value in parcellation_values)"
            )
        parcel_outline = matlab.logical(parcel_outline.tolist())

    outline_color = matlab.double(np.asarray(outline_color, dtype=float).tolist())
    background_color = matlab.double(
        np.asarray(background_color, dtype=float).tolist()
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
        parcel_alpha,
        parcel_outline,
        outline_color,
        float(outline_width),
        background_color,
        nargout=0
    )

