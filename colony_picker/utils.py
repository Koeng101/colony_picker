from scipy.spatial.transform import Rotation
import cv2
import numpy as np


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


def rotate_image_card_3d(
        img: np.ndarray,
        img_size: np.ndarray,
        out_size: np.ndarray,
        rot_mat: np.ndarray,
        center: np.ndarray,
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
        rot_mat (np.ndarray): A 3x3 rotation matrix to rotate the
            card.
        center (np.ndarray): The position in 3d space of the card.
        focal_length (float): The focal length of the camera viewing
            the card.

    Returns:
        np.ndarray: The resulting rotated image.
    """
    points = np.array([
        [0, 0, center[2]],
        [img_size[0], 0, center[2]],
        [img_size[0], img_size[1], center[2]],
        [0, img_size[1], center[2]]
    ])

    # Rotate around the center of the tag
    point_centroid = np.mean(points, axis=0)
    point_centroid[:2] += center[:2]
    points_centered = points - point_centroid

    points_rot = np.matmul(rot_mat, points_centered.T).T + point_centroid

    # Projection matrix
    proj_mat = np.zeros((3, 4), dtype=np.float64)
    proj_mat[:2, :2] = np.eye(2)
    proj_mat[2, 1] = 1 / focal_length
    proj_mat *= focal_length

    # Project original points and rotated points
    orig_points_project = np.matmul(proj_mat, augment_vec(points).T).T
    rot_points_project = np.matmul(proj_mat, augment_vec(points_rot).T).T

    # Create transformation matrix for perspective warping
    persp_mat = cv2.getPerspectiveTransform(
        orig_points_project[:, :2].astype(np.float32),
        rot_points_project[:, :2].astype(np.float32))
    res = cv2.warpPerspective(
        img, persp_mat, (out_size[0], out_size[1]), borderValue=255)
    return res
