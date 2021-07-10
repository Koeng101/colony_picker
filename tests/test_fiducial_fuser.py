from colony_picker.fiducial_fuser import Camera, Fiducial, FiducialFuser
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from colony_picker.utils import augment_vec, rotate_image_card_3d


class TestFiducialFuser(unittest.TestCase):
    def setUp(self):
        # First, let's make a few fiducials to detect using
        # the fiducial creator from opencv.
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        focal_length = 500
        tags = []
        ids = [1, 3, 12, 15]
        sizes = [100, 150, 125, 225]
        for id, size in zip(ids, sizes):
            tags.append(cv2.aruco.drawMarker(tag_dict, id, size))

        img = np.full((1000, 1000), 1.0)
        pos_list = [
            [0, 0, 0],
            [200, 50, 0],
            [500, 200, 0],
            [600, 500, 0]
        ]
        for tag, pos in zip(tags, pos_list):
            img[pos[0]:pos[0] + tag.shape[0], pos[1]:pos[1] + tag.shape[1]] = tag

        r_mat = Rotation.from_euler(
            'ZYX', np.array([0, np.pi / 4, 0])).as_matrix()
        rot_img = rotate_image_card_3d(
            img, (1, 1), (1000, 1000), r_mat, focal_length)
        self.test_rotated_image = rot_img
        self.fiducial_ids = ids
        self.tag_positions = np.array(pos_list)

    def test_fiducial_fuser(self):
        tags_ids = list(range(12))
        tag_xyz_1 = [[-27.25, 25 + 50 * (5 - i), 3.5] for i in range(6)]
        tag_xyz_2 = [[25 + + 50 * i, -27.25, 3.5] for i in range(6)]
        tag_pos = np.concatenate([tag_xyz_1, tag_xyz_2], axis=0)

        tag_trans_mat = np.tile(np.eye(4), (12, 1, 1))
        tag_trans_mat[:, :3, 3] = tag_pos

        fiducials = [Fiducial(id, trans, 32) for id, trans in zip(tags_ids, tag_trans_mat)]
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        fuser = FiducialFuser(fiducials, tag_dict)
        cap = cv2.VideoCapture('/home/maxschommer/Downloads/VID_20210705_144625216.mp4')
        _, frame = cap.read()
        f = 1600
        cam = Camera(f, f, frame.shape[0] / 2, frame.shape[1] / 2)

        for i in range(100):
            _, frame = cap.read()
            corners, ids, _ = cv2.aruco.detectMarkers(frame, tag_dict)
            pose = fuser.estimate_pose(corners, ids, cam)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            r_vec, _ = cv2.Rodrigues(pose[:3, :3])
            t_vec = pose[:3, 3]
            cv2.aruco.drawAxis(
                frame, cam.matrix.astype(np.float32),
                cam.dist_coefficients.astype(np.float32),
                r_vec, t_vec, 200)

            cv2.imshow("frame", frame)
            cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()
