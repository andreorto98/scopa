import unittest
import cv2

from scopa.training_data import prova
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa


class TestCore(unittest.TestCase):
    def test_prova(self):
        img3 = cv2.imread('tests/test_img/get_cards3.jpg',cv2.IMREAD_COLOR)
        a = len(prova(img3))
        self.assertEqual(a, 3)


if __name__ == '__main__':
    unittest.main()
