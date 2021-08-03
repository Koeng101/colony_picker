from colony_picker.fiducial_fuser import Camera, Fiducial, FiducialFuser
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class TestFiducialFuser(unittest.TestCase):

    def test_fiducial_fuser(self):
        """Test the fiducial fuser in a somewhat ridged way, by using known
        coordinates from the pose and comparing the result of a processed
        test frame.
        """
        target_pose = np.array(
            [[7.80452371e-01, 6.23116195e-01, -5.11889122e-02, -2.39836456e+02],
             [3.43159348e-01, -4.95364964e-01, -7.98032105e-01, -5.39769888e+00],
                [-5.22623897e-01, 6.05260134e-01, -6.00436866e-01, 7.19116577e+02],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

        tags_ids = list(range(12))
        tag_xyz_1 = [[-27.25, 25 + 50 * (5 - i), 3.5] for i in range(6)]
        tag_xyz_2 = [[25 + + 50 * i, -27.25, 3.5] for i in range(6)]
        tag_pos = np.concatenate([tag_xyz_1, tag_xyz_2], axis=0)

        tag_trans_mat = np.tile(np.eye(4), (12, 1, 1))
        tag_trans_mat[:, :3, 3] = tag_pos

        fiducials = [Fiducial(id, trans, 32)
                     for id, trans in zip(tags_ids, tag_trans_mat)]
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        fuser = FiducialFuser(fiducials, tag_dict)

        frame = cv2.imread("test_data/tag_board.png")

        cam = Camera.from_matrix(np.array([
            [1.86252313e+03, 0.00000000e+00, 9.42534164e+02],
            [0.00000000e+00, 1.86354739e+03, 5.38226928e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

        corners, ids, _ = cv2.aruco.detectMarkers(frame, tag_dict)

        corners = np.stack(corners)
        ids = np.squeeze(ids)
        pose = fuser.estimate_pose(corners, ids, cam)

        self.assertTrue(np.allclose(pose, target_pose))


if __name__ == "__main__":
    unittest.main()
