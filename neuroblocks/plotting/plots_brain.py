"""
Functions to plot data in brain maps.

plot_surface_from_parcellation makes use of the software Surf Ice (for which we must
either pass its directory or have it already stored in our home) to map the values
from a parcellation into a 3D surface map of the brain.

plot_4_views_surface_from_parcellation generates a 4 image figure with the lateral
and medial views of the left and right hemispheres, overlaying the parcellated values.

parcellation_to_brain_heatmap is a simplified function that takes a parcellation and
plots it as a heatmap in a 3-img display.

"""

import os
import nibabel
import subprocess

from PIL import Image
from pathlib import Path
from nilearn.plotting import plot_stat_map
from ..data_preparation.parcellation_tools import parcellation_to_volume


def parcellation_to_brain_heatmap(
    data_parcellated, parcellation_type, output_path, **kwargs_stat_map
):
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
    data_img = parcellation_to_volume(data_parcellated, parcellation_type)

    # Display the map on the MNI152 template
    plot_stat_map(data_img, output_file=output_path, **kwargs_stat_map)


def plot_surface_from_parcellation(
    parcellation_values,
    dir_results,
    filename_root,
    parcellation_str,
    cmap="Red-Yellow",
    vminmax=None,
    view="lateral",
    hemisphere="right",
    surf_ice_dir=None,
    show_colorbar=False,
    colorbar_position=1
):
    # Get volume image from parcellation
    vol_img = parcellation_to_volume(parcellation_values, parcellation_str)

    # Store it so that it can be later plotted
    nibabel.save(vol_img, dir_results / "tmp.nii.gz")

    # Then generate the script to run with Surf Ice
    surfice_script = (
        f"import gl\ngl.resetdefaults()\n"
        f"gl.meshload('BrainMesh_ICBM152_smoothed.mz3')\n"
        f"gl.overlayload(r'{str(dir_results / 'tmp.nii.gz')}')\n"
        f"gl.overlaytransparencyonbackground(25)\n"
        f"gl.cameradistance(0.7)\n"
        f"gl.meshcurv()\n"
    )
    if vminmax is not None:
        if len(vminmax) == 2:
            vmin, vmax = vminmax
            surfice_script += f"gl.overlayminmax(1, {vmin}, {vmax})\n"
        else:
            raise ValueError("vminmax must be a tuple (vmin, vmax)")
    if show_colorbar:
        surfice_script += f"gl.colorbarvisible(1)\n"
        surfice_script += f"gl.colorbarposition({colorbar_position})\n"

    else:
        surfice_script += "gl.colorbarvisible(0)\n"

    if cmap:
        surfice_script += f"gl.overlaycolorname(1, '{cmap}')\n"

    # Adapt camera to show the desired view
    if view == "lateral":  # No need to clip here
        if hemisphere == "right":  # We must set camera to right
            surfice_script += "gl.viewsagittal(0)\n"
        elif hemisphere == "left":  # Camera set to the left
            surfice_script += "gl.viewsagittal(1)\n"
        else:
            raise ValueError("hemisphere must be 'right' or 'left'")

    elif view == "medial":  # Now we need to clip, looking from inside
        if hemisphere == "right":  # Now we observe from left and clip
            surfice_script += "gl.viewsagittal(1)\n"
            surfice_script += "gl.clipazimuthelevation(0.5, 90, 0)\n"

        elif hemisphere == "left":  # Now we observe from right and clip
            surfice_script += "gl.viewsagittal(0)\n"
            surfice_script += "gl.clipazimuthelevation(0.5, 270, 0)\n"

        else:
            raise ValueError("hemisphere must be 'right' or 'left'")
    else:
        raise ValueError("view must be 'lateral' or 'medial'")

    # Save the image
    filename = f"{filename_root}_{view}_{hemisphere}.png"
    surfice_script += f"gl.savebmp(r'{str(dir_results / filename)}')\nquit()"
    print(surfice_script)
    # Run the Surf Ice with subprocess
    if surf_ice_dir is None:
        surf_ice_dir = Path.home() / "Surf_Ice"
    subprocess.call([str(surf_ice_dir) + "/surfice", "-n", "-S", str(surfice_script)])

    # Finally, remove the temporal file
    os.remove(dir_results / "tmp.nii.gz")


def merge_vertical(image_paths, output_path):
    # Load images as RGBA
    images = [Image.open(p).convert("RGBA") for p in image_paths]

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    total_height = sum(heights)
    max_width = max(widths)

    # IMPORTANT: use RGBA canvas
    merged = Image.new("RGBA", (max_width, total_height), (0, 0, 0, 0))

    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset), mask=img)  # mask keeps transparency
        y_offset += img.height

    merged.save(output_path)


def plot_4_views_surface_from_parcellation(
    parcellation_values,
    dir_results,
    filename_root,
    parcellation_str,
    cmap=None,
    vmin=None,
    vmax=None,
    surfice_dir=None,
):
    hemis = ["left", "left", "right", "right"]
    views = ["lateral", "medial", "lateral", "medial"]

    img_paths = []
    for i, (hemi, view) in enumerate(zip(hemis, views)):
        print(f"Plotting {hemi} Hemisphere - {view} View")
        show_colorbar = False if i != 3 else True  # Colorbar at the bottom
        plot_surface_from_parcellation(
            parcellation_values,
            dir_results,
            f"{filename_root}_{i}",
            parcellation_str,
            cmap,
            vminmax=(vmin, vmax),
            view=view,
            hemisphere=hemi,
            surf_ice_dir=surfice_dir,
            show_colorbar=show_colorbar,
            colorbar_position=1,
        )
        img_paths.append(dir_results / f"{filename_root}_{i}_{view}_{hemi}.png")

    # Then, we concatenate them one on top of the other and save in a single file
    merge_vertical(img_paths, dir_results / f"{filename_root}_4views.png")

    # Consider deleting the remaining img_paths
    for img_path in img_paths:
        os.remove(img_path)
