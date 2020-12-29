import unittest
import cv2
import numpy as np

from scopa.training_data import transform_img, transform_img_args, n_card_to_string, n_card_to_number
from scopa.utilities import show_image
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa



class TestCore(unittest.TestCase):
    def test_transf_and_transf_args(self):
        img = cv2.imread('tests/test_img/pic_0.jpg',cv2.IMREAD_COLOR)
        img = transform_img(*transform_img_args(img, 500))
        check = transform_img_args(img, 500)[1:]     #(0.0, 1, (1, 0))
        self.assertAlmostEqual(check[0], 0., delta = 0.1)
        self.assertAlmostEqual(check[2][0], 0., delta = 2.)
        self.assertAlmostEqual(check[2][1], 0., delta = 2.)

        img = transform_img(img, angle = 30, scale = 1, tr = (20,40))
        correct_angle = -30
        correct_tr = (-20*np.cos(np.pi/6)-40*np.sin(np.pi/6), 20*np.sin(np.pi/6)-40*np.cos(np.pi/6))
        check = transform_img_args(img, 500)[1:]   #(-30.03, 1, (-36, -25))
        self.assertAlmostEqual(check[0], correct_angle, delta = 0.1)
        self.assertAlmostEqual(check[2][0], correct_tr[0], delta = 2.)
        self.assertAlmostEqual(check[2][1], correct_tr[1], delta = 2.)

    def test_n_card_to_string(self):
        self.assertEqual(n_card_to_string(17), '7_diamonds(quadri)')

if __name__ == '__main__':
    unittest.main()
