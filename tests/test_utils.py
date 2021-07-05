import unittest
import cv2
from colony_picker.utils import generate_fiducials


class TestUtils(unittest.TestCase):
    def test_generate_fiducials(self):
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        generate_fiducials(tag_dict, list(range(12)), 32, "test_tags", True)


if __name__ == "__main__":
    unittest.main()
