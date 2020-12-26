'''Test for the utilities.py file
'''

import unittest
import cv2
import os

from scopa.utilities import get_cards
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa


class TestCore(unittest.TestCase):
    def test_get_cards(self):
        img0 = cv2.imread('tests/test_img/get_cards0.jpg',cv2.IMREAD_COLOR)
        lenght = len(get_cards(img0, 50))
        self.assertEqual(lenght, 0, f'get_cards test failed: founded {lenght}, instead of 0')
        img3 = cv2.imread('tests/test_img/get_cards3.jpg',cv2.IMREAD_COLOR)
        lenght = len(get_cards(img3, 50))
        self.assertEqual(lenght, 3, f'get_cards test failed: founded {lenght}, instead of 3')



if __name__ == '__main__':
    unittest.main()
