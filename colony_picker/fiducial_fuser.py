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

    @property
    def matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    @property
    def dist_coefficients(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3, self.k4,
                         self.k5, self.k6, self.s1, self.s2, self.s3, self.s4])


class Fiducial():
    def __init__(self, id: int, transform: np.ndarray, size: float):
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
        self.fiducials = fiducials

        all_points = np.concatenate([f.points_3d for f in fiducials], axis=0)
        self.points = all_points
        self.ids = np.array([f.id for f in fiducials], dtype=np.int16)

        board_points = np.reshape(self.points, (len(self.ids), -1, 3)).astype(np.float32)
        self.board = cv2.aruco.Board_create(board_points, dictionary, self.ids)

    def estimate_pose(self, marker_corners, marker_ids, camera: Camera):
        _, r_vec, t_vec = cv2.aruco.estimatePoseBoard(marker_corners,
                                                      marker_ids,
                                                      self.board,
                                                      camera.matrix,
                                                      camera.dist_coefficients,
                                                      np.zeros(3, dtype=np.float32),
                                                      np.zeros(3, dtype=np.float32)
                                                      )

        r_mat, _ = cv2.Rodrigues(r_vec)
        transform = np.eye(4)
        transform[:3, :3] = r_mat
        transform[:3, 3] = t_vec
        return transform
