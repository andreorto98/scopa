from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2

from scopa.utilities import show_image, import_img, get_cards, get_area
from scopa.training_data import generate_cards, transform_img
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

suit_model = load_model('models/numb_model.h5')
    # suit_model takes an image 1x256x256
url = 'http://192.168.1.9:8080'
#print('insert a card')
#img = import_img(url=url, show_video=True)


inp = 'q'
while inp=='':
    card, correct = generate_cards(1, layers=1)
    show_image(card[0], 'generated card', 100)
    print(card.shape)
    pred = suit_model.predict(card)
    print(np.argmax(pred[0])+1)
    print(pred[0])
    inp = input('inp: ')
print('con le generate va alla grande!')

inp = ''
while inp=='':
    print('insert a card')
    img = import_img(url=url, show_video=True)
    card = get_cards(img, 17500, to_model=True)
    show_image(card[0], 'imported_card', 100)
    pred = suit_model.predict(card)
    print(np.argmax(pred[0])+1)
    print(pred[0])
    inp = input('inp: ')





'''
img = cv2.imread('img/7_clubs(fiori).jpg')
cards = get_cards(img, 17500*0.7, verbouse=False, to_model=True)
'''

#show_image(transform_img(cv2.imread('img/prova_rot.jpg'), 30))

'''
show_image(img)
cards = get_cards(img_r, 500, to_model=True)
show_image(cards[0])
'''


'''
for i in range(0,37):
    print(i)
    angle = 10*i
    img_r = transform_img(img, angle)
    cards = get_cards(img_r, 500, to_model=True)
'''




'''
upper_bound, card_area = get_upper_bound_and_area(500, margin=[30,30,0], url = url)
    # with img/2_harts_bounds_area.jpg returns upper_bound=[210, 206, 255], area = 17505.5
    # THE AREA MUST BE EQUALLY ILLUMINATED
print(upper_bound, card_area)



print('insert a card')
img = import_img(url=url, show_video=True)
cards = get_cards(img, 17500*0.7, verbouse=True, to_model=True, upper_bound = [190, 190, 255])

print(len(cards))
'''













'''
inp = ''
while inp=='':
    card, correct = generate_cards(1, layers=3)
    show_image(card[0], 'generated card', 100)
    pred = suit_model.predict(card)
    print(np.argmax(pred[0]))
    inp = input('inp: ')
'''
'''
#img = import_img(url, True, 'img/test_model')
img = cv2.imread('img/test_model.jpg')
cards = get_cards(img, 500, verbouse = True, to_model=True)
'''

#imgn = cv2.imread('img/7_clubs(fiori).jpg')
#cardsn = get_cards(imgn, 500, verbouse = True, to_model=True)



'''
pred = suit_model(cards)
for i in pred:
    print(np.argmax(i))
print(pred)

img_tr = cv2.imread('deck0/2_harts(cuori).jpg')
cards_tr = get_cards(img_tr, 500, verbouse = True, to_model=True)
show_image(cards_tr[0])
'''


'''
devi:
    - scegliere il freshold basandoti sul solo 2;
'''
