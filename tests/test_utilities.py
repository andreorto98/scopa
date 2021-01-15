'''Test for the utilities.py file
'''

import unittest
import cv2
from scopa.utilities import get_cards, show_image, convert_card, string_to_n_card, met1_to_cards, met2_to_cards, met3_to_cards
from tensorflow.keras.models import load_model, Model
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa


class TestCore(unittest.TestCase):
    def test_get_cards(self):
        img0 = cv2.imread('tests/test_img/get_cards0.jpg',cv2.IMREAD_COLOR)
        lenght = len(get_cards(img0, 50))
        self.assertEqual(lenght, 0, f'get_cards test failed: founded {lenght}, instead of 0')
        img3 = cv2.imread('tests/test_img/get_cards3.jpg',cv2.IMREAD_COLOR)
        cards = get_cards(img3, 50)
        lenght = len(cards)
        self.assertEqual(lenght, 3, f'get_cards test failed: founded {lenght}, instead of 3')
        self.assertEqual(cards.shape, (3,256,256))

    def test_convert_card(self):
        self.assertEqual(convert_card('7_diamonds'), (7,1))
        self.assertEqual(convert_card(17), (7,1))
        self.assertEqual(convert_card((7,1)), '7_diamonds')

    def test_string_to_n_card(self):
        self.assertEqual(string_to_n_card('7_diamonds'), 17)
        self.assertEqual(string_to_n_card('./path/7_diamonds.jpg'), 17)

    def test_met1_to_cards(self):
        numb_model = load_model('./scopa/models/numb_model.h5')
        suit_model = load_model('./scopa/models/suit_model.h5')
        card = cv2.imread('./scopa/deck_thr/7_diamonds.jpg', cv2.IMREAD_GRAYSCALE).reshape(1,256,256,1)
        self.assertEqual(met1_to_cards(card, numb_model, suit_model), [17])

    def test_met2_to_cards(self):
        all_model_1 = load_model('./scopa/models/all_model_1.h5')
        card = cv2.imread('./scopa/deck_thr/1_spades.jpg', cv2.IMREAD_GRAYSCALE).reshape(1,256,256,1)
        self.assertEqual(met2_to_cards(card, all_model_1), [31])

    def test_met3_to_cards(self):
        all_model_2 = load_model('./scopa/models/all_model_2.h5')
        card = cv2.imread('./scopa/deck_thr/5_diamonds.jpg', cv2.IMREAD_GRAYSCALE).reshape(1,256,256,1)
        self.assertEqual(met3_to_cards(card, all_model_2), [15])





if __name__ == '__main__':
    unittest.main()
