import logging
from typing import List
import numpy as np
import cv2
from numpy.lib.index_tricks import r_
import quaternion
from colony_picker.utils import augment_vec


class Camera():
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0, k2: float = 0, p1: float = 0, p2: float = 0,
                 k3: float = 0, k4: float = 0, k5: float = 0, k6: float = 0,
                 s1: float = 0, s2: float = 0, s3: float = 0, s4: float = 0):
        """A Camera object containing the camera parameters of a camera.

        Args:
            fx (float): The focal length in X (in pixels)
            fy (float): The focal length in Y (in pixels)
            cx (float): The X optical center of the camera (in pixels)
            cy (float): The Y optical center of the camera (in pixels)
            k1 (float, optional): Radial coefficient 1. Defaults to 0.
            k2 (float, optional): Radial coefficient 2. Defaults to 0.
            p1 (float, optional): Tangential coefficient 1. Defaults to 0.
            p2 (float, optional): Tangential coefficient 2. Defaults to 0.
            k3 (float, optional): Radial coefficient 3. Defaults to 0.
            k4 (float, optional): Radial coefficient 4. Defaults to 0.
            k5 (float, optional): Radial coefficient 5. Defaults to 0.
            k6 (float, optional): Radial coefficient 6. Defaults to 0.
            s1 (float, optional): Spherical coefficient 1. Defaults to 0.
            s2 (float, optional): Spherical coefficient 2. Defaults to 0.
            s3 (float, optional): Spherical coefficient 3. Defaults to 0.
            s4 (float, optional): Spherical coefficient 4. Defaults to 0.
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, *args, **kwargs) -> "Camera":
        """Construct a Camera from a camera matrix.

        Returns:
            Camera: The resulting Camera instance.
        """

        fx = matrix[0, 0]
        fy = matrix[1, 1]
        center = matrix[:2, 2]
        return cls(fx, fy, *center, *args, **kwargs)

    @property
    def matrix(self) -> np.ndarray:
        """The matrix representaion of the camera (does not include distortion
        coefficients).

        Returns:
            np.ndarray: A 3x3 camera matrix.
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    @property
    def dist_coefficients(self) -> np.ndarray:
        """A length 12 array of distortion coefficients.

        Returns:
            np.ndarray: The distortion coefficients.
        """
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3, self.k4,
                         self.k5, self.k6, self.s1, self.s2, self.s3, self.s4], dtype=np.float)


class Fiducial():
    def __init__(self, id: int, transform: np.ndarray, size: float):
        """Initialize a Fiducial instance. A fiducial has an ID and a transform,
        which is relative to it's origin coordinate system.

        Args:
            id (int): The id of the tag.
            transform (np.ndarray): A 4x4 transformation matrix representing the
                relative transform of the tag's parent coordinate system to the
                tag's coordinate system.
            size (float): The size (in mm) of the side of the tag. This is the
                size of the black border of the tag, and refers to the real
                world size of the tag (ie. the size of the tag printed out and
                viewed by the camera).
        """
        self.id = id
        self.transform = transform
        self.size = size

    @property
    def points_2d(self) -> np.ndarray:
        """The centered 2d points of the fiducial (in clockwise order starting
        from the top left).

        Returns:
            np.ndarray: A 4x2 array of 2d points.
        """
        points = [
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5]
        ]
        return np.array(points) * self.size

    @property
    def points_3d(self) -> np.ndarray:
        """The transformed 3D points of the fiducial, transformed by
        self.transform and in the same ordering as points_2d.

        Returns:
            np.ndarray: A 4x3 array of 3d points.
        """
        points = np.concatenate(
            [self.points_2d, np.tile([0, 1], (4, 1))], axis=1)
        return np.matmul(self.transform, points.T).T[:, :3]


class FiducialFuser():
    def __init__(self, fiducials: List[Fiducial],
                 dictionary: cv2.aruco_Dictionary):
        """Create a FiducialFuser instance using a list of Fiducials and
        a tag family dictionary.

        Args:
            fiducials (List[Fiducial]): A list of Fiducial objects, each of
                which have their own relative transform to the Fuser's
                coordinate system.
            dictionary (cv2.aruco_Dictionary): An OpenCV Aruco tag dictionary
                used to describe the tag family.
        """
        self.fiducials = fiducials

        all_points = np.concatenate([f.points_3d for f in fiducials], axis=0)
        self.points = all_points
        self.ids = np.array([f.id for f in fiducials], dtype=np.int16)

        board_points = np.reshape(
            self.points, (len(self.ids), -1, 3)).astype(np.float32)
        self.board = cv2.aruco.Board_create(board_points, dictionary, self.ids)

    def estimate_pose(self, marker_corners: np.ndarray, marker_ids: np.ndarray,
                      camera: Camera) -> np.ndarray:
        """Estimate the pose of the FiducialFuser from a set of markes

        Args:
            marker_corners (np.ndarray): An #Nx4x2 array of marker corners,
                where #N is the number of markers. The second and third
                dimensions of the array are the number of corners (4), and the
                image XY coordinates of the corners (2).
            marker_ids (np.ndarray): A length #N array of marker ids associated
                with each marker in the marker_corners array.
            camera (Camera): A Camera object representing the camera parameters
                of the camera which took the image which produced the marker
                corners. 

        Returns:
            np.ndarray: A 4x4 array representing the transform of the
                FiducialFuser estimated by the tags.
        """
        _, r_vec, t_vec = cv2.aruco.estimatePoseBoard(marker_corners,
                                                      marker_ids,
                                                      self.board,
                                                      camera.matrix,
                                                      camera.dist_coefficients,
                                                      np.zeros(
                                                          3, dtype=np.float32),
                                                      np.zeros(
                                                          3, dtype=np.float32)
                                                      )

        r_mat, _ = cv2.Rodrigues(r_vec)
        transform = np.eye(4)
        transform[:3, :3] = r_mat
        transform[:3, 3] = t_vec
        return transform
