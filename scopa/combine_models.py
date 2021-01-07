from tensorflow.keras.models import load_model, Model
from scipy import stats
import numpy as np
import cv2


from scopa.utilities import show_image, get_cards, import_img
from scopa.training_data import n_card_to_string, transform_img

#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

url = 'http://192.168.1.9:8080'
    # all the models take n images nx256x256
numb_model = load_model('models/numb_model.h5')             # model_number.ipynb
suit_model = load_model('models/suit_model.h5')             # model_suit.ipynb
all_model_1 = load_model('models/all_model_1.h5')           # card-rec1layer_alltogether.ipynb
    # these model returns an array of lenght n, with predictions in one hot notation

all_model_2 = load_model('models/all_model_2.h5')           # card-rec1layer_alltogether2.ipynb
    # this model return a list with the 2 predictions in one hot notation
'''
img = cv2.imread('img/7_clubs(fiori).jpg')
cards = get_cards(img, 17500, to_model=True)
pred_s = suit_model.predict(cards)
pred_n = numb_model.predict(cards)
pred_1 = all_model_1.predict(cards)
pred_2 = all_model_2.predict(cards)

print('pred_s')
print(pred_s)
print('***')
print('pred_n')
print(pred_n)
print('***')
print('pred_1')
print(pred_1)
print('***')
print('pred_2')
print(pred_2)
'''

# metodo 1: suit and number separately
def met1_to_cards(cards):
    pred_n = numb_model.predict(cards)
    pred_s = suit_model.predict(cards)
    return [np.argmax(pred_n[i])+1 + 10*np.argmax(pred_s[i]) for i in range(len(cards))]

def met2_to_cards(cards):
    pred_1 = all_model_1.predict(cards)
    return [np.argmax(pred_1[i])+1 for i in range(len(cards))]

def met3_to_cards(cards):
    [pred_2_n, pred2_s] = all_model_2.predict(cards)
    return [np.argmax(pred_2_n[i])+1 + 10*np.argmax(pred2_s[i]) for i in range(len(cards))]


def img_to_cards(img):
    cards = get_cards(img, 17500, to_model=True)
    print(f'Founded cards: {len(cards)}')
    #for card in cards:
    #    show_image(card)
    cards_r = np.array([met1_to_cards(cards), met2_to_cards(cards), met3_to_cards(cards)]).T
    #print(cards_r)
    c = []
    for card in cards_r:
        if len(np.unique(card)) == 3:
            print(f'Ambigous Card: {card}')
        else:
            mode, freq = stats.mode(card)
            print(n_card_to_string(mode[0]) + f' ({freq[0]})')
            c.append(mode[0])
    return c

            # stats.mode returns an object like: ModeResult(mode=array([27]), count=array([3]))

#print(img_to_cards(cv2.imread('img/7_clubs(fiori).jpg')))


inp = ''
while inp=='':
    print('insert cards')
    img = import_img(url=url, show_video=True)
    original = sorted(img_to_cards(img))

    img = transform_img(img, angle = 90)
    rotated = sorted(img_to_cards(img))
    print(original)
    print(rotated)
    print(len(original), len(rotated))

    amb_cards = [(original[i], rotated[i]) for i in range(len(original)) if original[i]!=rotated[i]]

    print(amb_cards)
    #attento ad ambigous card

    inp = input('inp: ')



'''
- se una carta è troppo vicina al bordo si rompe tutto (in get_cards imponi out = 256x256 shaped)
- vedi se riesci a migliorare un po i successi (a volte sbaglia... potresti fare un processo parallelo con l'immagine ruotata di 90 gradi e vedere cosa succede)
'''
