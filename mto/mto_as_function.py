import argparse
from ctypes import c_double, c_float
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from mto import _ctype_classes as ct
from mto.mto_as_script import build_max_tree
from mto.postprocessing import get_image_parameters, relabel_segments
from mto.preprocessing import preprocess_image
from mto.tree_filtering import (
    filter_tree,
    get_c_significant_nodes,
    init_double_filtering,
)


def assemble_arguments(*args, **kwargs):
    # Create a Namespace object with the provided arguments
    namespace = argparse.Namespace(*args, **kwargs)
    return namespace


def mto(
    path,
    exts=None,
    fits_out=None,
    par_out=None,
    soft_bias=0,
    gain=-1,
    bg_mean=None,
    bg_variance=-1,
    alpha=1e-6,
    move_factor=0.5,
    min_distance=0.0,
    verbosity=0,
    overwrite=False,
):
    pars = assemble_arguments(
        par_out=par_out,
        soft_bias=soft_bias,
        gain=gain,
        bg_mean=bg_mean,
        bg_variance=bg_variance,
        alpha=alpha,
        move_factor=move_factor,
        min_distance=min_distance,
        verbosity=verbosity,
    )
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    if exts is not None:
        with fits.open(path) as hdul:
            if isinstance(hdul[0], fits.PrimaryHDU):
                primary_hdr = hdul[0].header
            else:
                primary_hdr = None
            imgs = []
            wcses = []
            for ext in exts:
                if isinstance(ext, str):
                    if ext not in hdul:
                        raise ValueError(f"Extension {ext} not found in {path}.")
                if isinstance(ext, int):
                    if ext >= len(hdul):
                        raise ValueError(f"Extension {ext} not found in {path}.")
                imgs.append(hdul[ext].data)
                wcses.append(WCS(hdul[ext].header))
    else:
        with fits.open(path) as hdul:
            if isinstance(hdul[0], fits.PrimaryHDU):
                primary_hdr = hdul[0].header
            else:
                primary_hdr = None
            imgs = []
            wcses = []
            for hdu in hdul:
                if (
                    isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU))
                    and hdu.data is not None
                ):
                    imgs.append(hdu.data)
                    wcses.append(WCS(hdu.header))
    if len(imgs) > 0:
        with Pool(len(imgs)) as pool:
            res = pool.starmap(mto_pipeline_per_img, zip(imgs, [pars] * len(imgs)))
            segmaps, src_pars = zip(*res)
    else:
        segmaps, src_pars = mto_pipeline_per_img(imgs[0], pars)
        segmaps = [segmaps]
        src_pars = [src_pars]
    for segmap in segmaps:
        segmap += 1  # Ensure segment IDs start from 1
    # write segmaps to fits
    if fits_out is not None:
        hdus_out = [fits.PrimaryHDU(header=primary_hdr)]
        for segmap, wcs in zip(segmaps, wcses):
            hdr = wcs.to_header()
            i16 = np.iinfo(np.int16)
            i8 = np.iinfo(np.int8)
            can_i8 = (segmap.min() >= i8.min) and (segmap.max() <= i8.max)
            if can_i8:
                segmap = segmap.astype(np.int8)
            else:
                can_i16 = (segmap.min() >= i16.min) and (segmap.max() <= i16.max)
                if can_i16:
                    segmap = segmap.astype(np.int16)
            hdus_out.append(fits.ImageHDU(segmap, header=hdr))
        fits.HDUList(hdus_out).writeto(fits_out, overwrite=overwrite)
    # write src_pars to csv
    if par_out is not None:
        par_out = Path(par_out).expanduser()
        src_pars = np.concatenate(src_pars)
        np.savetxt(par_out, src_pars, delimiter=",", fmt="%s")
    return segmaps


def mto_pipeline_per_img(img, pars):
    pars.d_type = c_float
    if np.issubdtype(img.dtype, np.float64):
        pars.d_type = c_double
        init_double_filtering(pars)

    # Initialise CTypes classes
    ct.init_classes(pars.d_type)

    processed_img = preprocess_image(img, pars, n=2)

    # Build a max tree
    mt = build_max_tree(processed_img, pars)

    # Filter the tree and find objects
    id_map, sig_ancs = filter_tree(mt, processed_img, pars)

    # Relabel objects for clearer visualization
    id_map = relabel_segments(id_map, shuffle_labels=True)

    # Generate output files
    segmap = id_map.reshape(img.shape)
    object_ids = id_map.ravel()
    if pars.par_out is not None:
        # Mask NANs for parameter calculations
        img_ma = np.ma.array(img, mask=np.isnan(img))
        src_pars = get_image_parameters(img_ma, object_ids, sig_ancs, pars)
    else:
        src_pars = ""
    return segmap, src_pars
