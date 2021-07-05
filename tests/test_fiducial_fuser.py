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
        pass


if __name__ == "__main__":
    unittest.main()
