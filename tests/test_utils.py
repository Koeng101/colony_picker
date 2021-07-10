import unittest
import cv2
from colony_picker.utils import generate_fiducials, calilbrate_camera


class TestUtils(unittest.TestCase):
    def test_generate_fiducials(self):
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        generate_fiducials(tag_dict, list(range(12)), 32, "test_tags", True)

    def test_calibrate_camera(self):
        square_size = 19.3  # millimeters
        x_squares = 9
        y_squares = 6
        vid_file = ""
        calilbrate_camera()


if __name__ == "__main__":
    unittest.main()
