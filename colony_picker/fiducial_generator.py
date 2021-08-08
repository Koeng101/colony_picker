
import json
import numpy as np
import cv2
import plac


ARUCO_DICT_MAPPER = {
    "4X4_50": (4, 50),
    "4X4_100": (4, 100),
    "4X4_250": (4, 250),
    "4X4_1000": (4, 1000),
    "5X5_50": (5, 50),
    "5X5_100": (5, 100),
    "5X5_250": (5, 250),
    "5X5_1000": (5, 1000),
    "6X6_50": (6, 50),
    "6X6_100": (6, 100),
    "6X6_250": (6, 250),
    "6X6_1000": (6, 1000),
    "7X7_50": (7, 50),
    "7X7_100": (5, 100),
    "7X7_250": (7, 250),
    "7X7_1000": (7, 1000),
}


def generate_fiducials(
        tag_dictionary: cv2.aruco_Dictionary,
        ids: np.ndarray,
        size: float,
        include_white_border: bool = False,
        dpi: float = 300):
    """Generate a set of Aruco tag fiducial images and save them as files.

    Args:
        tag_dictionary (cv2.aruco_Dictionary): The aruco tag dictionary
            describing the tag family.
        ids (np.ndarray): A length #N vector of integer tag ids to create
            fiducial images for. The ids should be valid for the provided tag
            dictionary.
        size (float): The side length of the fiducial square in mm (at the
            provided dpi).
        include_white_border (bool, optional): If True, then add a white border
            which is the same width as the tag squares, as well as a single
            pixel black border (to aid in cutting out the tags). This is useful
            if placing the tags on a dark object, so that sufficient contract
            is provided by the tags themselves. Defaults to False.
        dpi (float, optional): The pixels per inch of the resulting image.
            Defaults to 300.
    """
    pixels = int(dpi * (size / 25.4))  # Convert to mm
    res = []
    for id in ids:

        tag = cv2.aruco.drawMarker(tag_dictionary, id, pixels)
        if include_white_border:
            num_squares = tag_dictionary.markerSize + 2
            square_size = int(pixels / num_squares)
            # One additional pixel is left on each side or a "black" border to
            # make tag easy to cut out.
            new_pixels = pixels + 2 * square_size + 2

            border = np.zeros((new_pixels, new_pixels))
            border[1:-1, 1:-1] = 255
            border[square_size:square_size + pixels,
                   square_size:square_size + pixels] = tag
            tag = border
        res.append(tag)
    return res


@plac.pos(
    'tag_dictionary', help='The aruco tag dictionary describing the tag'
    ' family.', type=str, choices=ARUCO_DICT_MAPPER.keys()
)
@plac.pos(
    'ids', help='The tag IDS to generate. The maximum ID number can be at most'
    ' the maximal size of the tag dictionary used. For example, if a 4X4_50'
    ' dictionary is used, the maximal id can be 49. This should be a string'
    ' formatted as "[id0, id1, ..., idn]".', type=str
)
@plac.opt(
    'size', help='The size of the fiducial (width of the square) in mm.'
    ' Specifically, this is the width of the black border around the fiducial'
    ' and should be the same as input into the fiducial identification in order'
    ' to get real-world dimensions to match the fiducial readings.', type=float,
    abbrev='s'
)
@plac.opt(
    'base_file_name', help='The base name of the image files generated. Image'
    ' files will be of the format: "[base_file_name]_[tag_id].png', type=str,
    abbrev='bfn'
)
@plac.flg(
    'include_white_border', help='If provided, then add a white border which is'
    ' the same width as the tag squares, as well as a single pixel black border'
    ' (to aid in cutting out the tags). This is useful if placing the tags on a'
    ' dark object, so that sufficient contract is provided by the tags'
    ' themselves.', abbrev='iwb'
)
@plac.opt(
    'dpi', help='The pixels per inch of the resulting image.'
)
def fiducial_generator(
    tag_dictionary: cv2.aruco_Dictionary,
    ids: str,
    size: float = 100,
    base_file_name: str = "tag",
    include_white_border: bool = False,
    dpi: float = 300
):
    tag_n_size, num_tags = ARUCO_DICT_MAPPER[tag_dictionary]
    td = cv2.aruco.custom_dictionary(
        num_tags, tag_n_size)
    ids = json.loads(ids)
    tags = generate_fiducials(td, np.asarray(
        ids), size, include_white_border, dpi)

    for id, tag in zip(ids, tags):
        cv2.imwrite(f"{base_file_name}_{id}.png", tag)


if __name__ == "__main__":
    plac.call(fiducial_generator)
