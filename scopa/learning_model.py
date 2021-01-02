from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys
import time

#from math import *

from scopa.utilities import show_image, import_img, get_cards
from scopa.training_data import n_card_to_string, generate_cards

#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

n_samples = 20000

# con 10000 impiega 70s per laprima parte e usa circa 10 GB

print('start time')
start = time.time()

# Generate training data
train_data, target = generate_cards(n_samples, 3, False)

print('cards_generated')
print("--- %s seconds ---" % (time.time() - start))

print(f'Dimension of train_data (MB): {sys.getsizeof(train_data)/1e6:.0f}', train_data.shape)

# Reduce dimensionality of the image
train_data = np.array(MaxPooling2D(pool_size=(2,2), strides=None,
                      data_format='channels_last')(train_data))
    # pooled.shape = (n_samples, 360, 360, 3)
    # output_shape = (input_shape - pool_size + 1) / strides)

print('end of max_pool')
print("--- %s seconds ---" % (time.time() - start))
# it takes 6.6 seconds for layers=3, n_samples=1000

print(f'Dimension of pooled train_data (MB): {sys.getsizeof(train_data)/1e6:.0f}')


'''     MODEL1:
        Each image contains one card.
        The card is recognised looking separately to the suit and to the carder

        lo metteremo dentro una funzione train_model1()
'''

def card_to_suit(card):
    if card<11:
        return [1.,0.,0.,0.]
    elif card<21:
        return [0.,1.,0.,0.]
    elif card<31:
        return [0.,0.,1.,0.]
    else:
        return [0.,0.,0.,1.]

model1 = True
if model1:
    # creating the ground truth labels
    labels = np.array([card_to_suit(card) for card in target])

    # building the model: suit
    inputs = Input(shape=train_data[0].shape)
    conv = Conv2D(5, (10,10), activation = 'relu')(inputs)
    pool = MaxPooling2D(pool_size=(2,2))(conv)
    conv = Conv2D(4, (5,5), activation = 'relu')(pool)
    pool = MaxPooling2D(pool_size=(3,3))(conv)
    conv = Conv2D(3, (4,4), activation = 'relu')(pool)
    pool = MaxPooling2D(pool_size=(3,3))(conv)

    flatten = Flatten()(pool)
    dense = Dense(32, activation='relu')(flatten)
    outputs = Dense(4,activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['categorical_accuracy'])

    model.summary()
    history = model.fit(train_data, labels, validation_split=0.8, epochs= 15, verbose=1)
    '''
    plt.plot(history.history["categorical_accuracy"], label='Accuracy')
    plt.plot(history.history["val_categorical_accuracy"], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    '''
    print('end of training')
    print("--- %s seconds ---" % (time.time() - start))

    test_data, target = generate_cards(n_samples, 3, False)
    test_data = np.array(MaxPooling2D(pool_size=(2,2), strides=None,
                          data_format='channels_last')(test_data))

    test_pred = model.predict(test_data)
    test_pred = np.array([np.argmax(test_pred[i]) for i in range(len(test_pred))])
    test_label = np.array([card_to_suit(card) for card in target])
    test_label = np.array([np.argmax(test_label[i]) for i in range(len(test_label))])

    from sklearn import metrics

    print(f'Accuracy test set:\t{metrics.accuracy_score(test_label, test_pred):.1%}')
    print('val_categorical_accuracy')
    print(history.history["val_categorical_accuracy"])
    print('categorical_accuracy')
    print(history.history["categorical_accuracy"])

    print('----------------------------------------------')
