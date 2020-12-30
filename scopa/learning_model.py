from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import cv2

#from math import *

from scopa.utilities import show_image, import_img, get_cards
from scopa.training_data import n_card_to_string, generate_card

#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

n_samples = 50

# Generate training data
training_data = [generate_card(3) for i in range(n_samples)]
tr_input = [elem[0] for elem in training_data]
tr_card = [elem[1] for elem in training_data]

# Reduce dimensionality of the image
show_image(tr_input[0])
pooled = MaxPooling2D(pool_size=(2,2), strides=None, data_format='channels_last')(tr_input)
    # output_shape = (input_shape - pool_size + 1) / strides)
show_image(tr_input[0])





'''     MODEL1:
        Each image contains one card.
        The card is recognised looking separately to the suit and to the number

        lo metteremo dentro una funzione train_model1()
'''
