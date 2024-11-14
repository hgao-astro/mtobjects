"""High level processes for MTObjects."""

from ctypes import c_double, c_float

import numpy as np

from mto import _ctype_classes as ct
from mto import maxtree
from mto.io_mto import (
    generate_image,
    generate_parameters,
    make_parser,
    read_fits_file,
)
from mto.postprocessing import relabel_segments
from mto.preprocessing import preprocess_image
from mto.tree_filtering import (
    filter_tree,
    get_c_significant_nodes,
    init_double_filtering,
)
from mto.utils import time_function


def setup():
    """Read in a file and parameters; run initialisation functions."""

    # Parse command line arguments
    p = make_parser().parse_args()

    # Warn if using default soft bias
    if p.soft_bias is None:
        p.soft_bias = 0.0

    img = read_fits_file(p.filename)

    if p.verbosity:
        print("\n---Image dimensions---")
        print("Height = ", img.shape[0])
        print("Width = ", img.shape[1])
        print("Size = ", img.size)

    # Set the pixel type based on the type in the image
    p.d_type = c_float
    if np.issubdtype(img.dtype, np.float64):
        p.d_type = c_double
        init_double_filtering(p)

    # Initialise CTypes classes
    ct.init_classes(p.d_type)

    return img, p


def max_tree_timed(img, params, maxtree_class):
    """Build and return a maxtree of a given class"""
    if params.verbosity:
        print("\n---Building Maxtree---")
    mt = maxtree_class(img, params.verbosity, params)
    mt.flood()
    return mt


def build_max_tree(img, params, maxtree_class=maxtree.OriginalMaxTree):
    return time_function(
        max_tree_timed,
        (img, params, maxtree_class),
        params.verbosity,
        "create max tree",
    )
