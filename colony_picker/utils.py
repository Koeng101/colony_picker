from typing import Any, Optional
import textwrap
from scipy.spatial.transform import Rotation
import random
import cv2
import numpy as np


def print_describe(name: str, item: Any):
    """Print an item with a description, where the item is properly indented.

    Args:
        name (str): The name of the item to print.
        item (Any): The item to print, indented by one tab.
    """
    print(f"{name}: ")
    print(textwrap.indent(str(item), "\t"))


def augment_vec(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Augment a vector by appending a row of ones along the last
    axis.

    Args:
        v (np.ndarray): The array like to augment, of shape (..., N).
        axis (int): The axis to augment.

    Returns:
        np.ndarray: The resulting augmented vector of shape (..., N+1)
            where elements along the last axis are ones.
    """
    return np.concatenate([v, np.ones((*v.shape[:-1], 1))], axis=1)


def project_points(points: np.ndarray, focal_length: float) -> np.ndarray:
    """Project a set of 3D points using the pinhole camera model.

    Args:
        points (np.ndarray): An #Nx3 array of points to project
        focal_length (float): The focal length of the camera.

    Returns:
        np.ndarray: An #Nx2 array of projected points.
    """
    proj_mat = np.zeros((3, 4), dtype=np.float64)
    proj_mat[:2, :2] = np.eye(2)
    proj_mat[2, 2] = 1 / focal_length
    proj_mat *= focal_length

    proj_points = np.matmul(proj_mat, augment_vec(points).T).T
    proj_points /= proj_points[:, 2][:, np.newaxis]
    return proj_points[:, :2]


def nd_width(v: np.ndarray, axis=0) -> np.ndarray:
    """Get the difference between the minimum and maximum of an array along an
    axis.

    Args:
        v (np.ndarray): An ndarray to get the width of.
        axis (int, optional): The axis to take the min and max values from.
            Defaults to 0.

    Returns:
        np.ndarray: An array of the (max - min) along an axis of the array.
    """
    return np.max(v, axis=axis) - np.min(v, axis=axis)


def pv_spheres_from_numpy(v: np.ndarray, radius: float) -> "pv.PolyData":
    """Generate a pv.PolyData with spherical glyphs on each point given by v
    with the given radius.

    Args:
        v (np.ndarray): An #Nx3 array of 3d points.
        radius (float): The radius of the spheres to generate.

    Returns:
        pv.PolyData: The resulting PolyData object with spherical glyphs.
    """
    import pyvista as pv

    pv_points = pv.PolyData(v)
    return pv_points.glyph(geom=pv.Sphere(radius=radius))


def pv_line_from_numpy(verts: np.ndarray,
                       edges: Optional[np.ndarray] = None) -> "pv.PolyData":
    """Produces a PyVista Polyline (pv.PolyData) object from numpy
    vertex and edge arrays.
    Args:
        verts (np.ndarray): Vertex array of #Vx3
        edges (Optional[np.ndarray]): Edge array of #Ex2. If not provided, then
            the edges are assumed to be consecutive points.
    Returns:
        pv.PolyData: PolyLine object created from the vertices and edges
    """
    import pyvista as pv
    poly = pv.PolyData()
    poly.points = verts
    if edges is None:
        idxs = np.arange(verts.shape[0])
        edges = np.stack([idxs[:-1], idxs[1:]], axis=1)
    cells = np.full((len(edges), 3), 2, dtype=np.int_)
    cells[:, 1:3] = edges
    poly.lines = cells
    return poly


def view_projection(points_3d: np.ndarray,
                    focal_length: float,
                    bounds_sphere_ratio: float = 500,
                    plotter: "pv.Plotter" = None) -> "pv.Plotter":
    """Visualize a projection given the 3D points to project and the 2D
    projected points, as well as the focal length. This function draws the
    projected points on the image plane (at distance focal_length from the
    origin) and negates the xy coordinates, and then draws the associated lines
    between the 3D points and the projected points.

    Args:
        points_3d (np.ndarray): An #Nx3 array of 3D points to project.
        focal_length (float): The focal length of the camera to use to generate
            the projection matrix with.
        bounds_sphere_ratio (float, optional): The ratio of the bounds of the
            points to the radius of the sphere to use for visualization.
            Defaults to 500.
        plotter (pv.Plotter): If provided, then apply the plotting operations
            on the provided plotter.
    """
    import pyvista as pv
    points_proj = project_points(points_3d, focal_length)
    points_proj_3d = -np.concatenate([points_proj, np.full(
        (points_proj.shape[0], 1), focal_length)], axis=1)
    all_points = np.concatenate([points_3d, points_proj_3d], axis=0)
    bounds_size = np.max(nd_width(all_points))

    pv_points = pv_spheres_from_numpy(
        all_points, bounds_size / bounds_sphere_ratio)
    edges = np.stack([np.arange(points_3d.shape[0]),
                      points_3d.shape[0] + np.arange(points_3d.shape[0])], axis=1)

    if plotter is None:
        plotter = pv.Plotter()
    plotter.add_mesh(pv_points)
    plotter.add_mesh(pv_line_from_numpy(all_points, edges))
    return plotter


def generate_fiducials(
        tag_dictionary: cv2.aruco_Dictionary,
        ids: np.ndarray,
        size: float,
        base_file_name: str,
        include_white_border: bool = False,
        dpi: float = 300):
    pixels = int(dpi * (size / 25.4))  # Convert to mm

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
        cv2.imwrite(f"{base_file_name}_{id}.png", tag)


def rotate_image_card_3d(
        img: np.ndarray,
        img_size: np.ndarray,
        out_size: np.ndarray,
        rot_mat: np.ndarray,
        focal_length: float) -> np.ndarray:
    """Rotate an image as if it were a card seen by a camera. This
    rotates the card by the given rotation matrix around the card's
    centroid. This function assumes the the card is in the xy plane
    at the start.

    This function uses the pinhole camera model.

    Args:
        img (np.ndarray): An #Nx#Mx(1, 3) array representing the image
            to rotate.
        img_size (np.ndarray): A (2,) vector representing the
            dimensions of the image card (width, height).
        out_size (np.ndarray): A (2,) vector representing the
            dimensions of the output image (width, height).
        rot_mat (np.ndarray): A 3x3 rotation matrix to rotate the
            card.
        focal_length (float): The focal length of the camera viewing
            the card.

    Returns:
        np.ndarray: The resulting rotated image.
    """
    points = np.array([
        [0, 0, 0],
        [img_size[0], 0, 0],
        [img_size[0], img_size[1], 0],
        [0, img_size[1], 0]
    ], dtype=np.float64)

    # Rotate around the center of the tag
    centroid = np.mean(points, axis=0)
    points -= centroid

    points_not_rot = points.copy()
    points_not_rot[:, 2] += focal_length

    points_rot = np.matmul(
        rot_mat, (points).T).T
    points_rot[:, 2] += focal_length

    # Project the points from the 3D point given as input
    orig_proj = project_points(points_not_rot, focal_length)
    rot_proj = project_points(points_rot, focal_length)

    # Scale the output points to be the output image size
    orig_proj *= img_size / np.max(nd_width(orig_proj, axis=0))
    rot_proj *= img_size / np.max(nd_width(rot_proj, axis=0))

    # Make the points "centered"
    orig_proj += np.array([out_size[0] / 2, out_size[1] / 2])
    rot_proj += np.array([out_size[0] / 2, out_size[1] / 2])

    # Create transformation matrix for perspective warping
    persp_mat = cv2.getPerspectiveTransform(
        orig_proj.astype(np.float32),
        rot_proj.astype(np.float32))

    res = cv2.warpPerspective(
        img, persp_mat, (*out_size,), borderValue=255)
    return res


def calibrate_camera(board_squares_x: int, board_squares_y: int,
                     board_square_size: float, frames: np.ndarray,
                     num_samples: int = 20):
    """Calibrate a camera using a know chess board and a sequence of images
    represented by a numpy array. If more images are present in the array than
    num_samples, then the images are sampled randomly to solve for the
    camera calibration, so the result may be non-deterministic.

    Args:
        board_squares_x (int): The number of squares on the chess board in the
            x direction. To count the squares, count both black and white
            squares.
        board_squares_y (int): The number of squares on the chess board in the
            y direction. To count the squares, count both black and white
            squares.
        board_square_size (float): The size of the squares on the physical
            board (in the units of your choice). The units you choose will
            affect the calibration matrices returned.
        frames (np.ndarray): An #Nx#Mx#O array of grayscale frames. There are #N
            frames, and the width and height are #O and #M respectively.
        num_samples (int, optional): The number of samples to take from the
            frames (which have detected the chessboard validly). Defaults to 20.

    Raises:
        ValueError: Frames must be grayscale (last dimension must not be 3)
        ValueError: Wrong number of dimensions detected.
        RuntimeError: No chessboard corners were detected.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A Tuple comprising:
            (np.ndarray): A 3x3 camera matrix
            (np.ndarray): A vector of length 5 of distortion coefficients. (k1,
                k2, p1, p2, k3)
    """
    board_points_x = board_squares_x - 1
    board_points_y = board_squares_y - 1

    termination_criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = np.zeros(
        (board_points_x * board_points_y, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:board_points_x,
                                    0:board_points_y].T.reshape(-1, 2) * board_square_size

    if frames.shape[-1] == 3:
        raise ValueError("Frames must be converted to gray.")
    if frames.ndim != 3:
        raise ValueError(f"Frames have {frames.ndim} dimensions. Expected 3.")

    all_corners = []
    for i, frame in enumerate(frames):
        ret, corners = cv2.findChessboardCorners(
            frame, (board_points_x, board_points_y),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners_refined = cv2.cornerSubPix(frame, corners,
                                               (11, 11), (-1, -1),
                                               termination_criteria)

            all_corners.append(corners_refined.reshape(-1, 2))

    if len(all_corners) == 0:
        raise RuntimeError("No corners were detected. The board_squares_x or y"
                           " may be incorrect, or the full chessboard might not be in"
                           " focus or in view for any frames.")

    if len(all_corners) > num_samples:
        samp_idxs = random.sample(range(len(all_corners)), num_samples)
        all_corners = [all_corners[i] for i in samp_idxs]
    object_points = [object_points] * len(all_corners)

    ret, cam_mat, dist, _, _ = cv2.calibrateCamera(object_points,
                                                   all_corners,
                                                   frames.shape[1::-1], None,
                                                   None)
    return cam_mat, np.squeeze(dist)
