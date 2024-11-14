from mto import mto_as_script

"""Example program - using original settings"""


def main():
    # Get the input image and parameters
    image, params = mto_as_script.setup()

    # Pre-process the image
    processed_image = mto_as_script.preprocess_image(image, params, n=2)

    # Build a max tree
    mt = mto_as_script.build_max_tree(processed_image, params)

    # Filter the tree and find objects
    id_map, sig_ancs = mto_as_script.filter_tree(mt, processed_image, params)

    # Relabel objects for clearer visualisation
    id_map = mto_as_script.relabel_segments(id_map, shuffle_labels=False)

    # Generate output files
    mto_as_script.generate_image(image, id_map, params)
    mto_as_script.generate_parameters(image, id_map, sig_ancs, params)
