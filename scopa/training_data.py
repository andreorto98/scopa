"""Generate useful images to train the NN
"""

import numpy as np
import cv2
import sys
import random

from scopa.utilities import show_image, import_img, get_cards
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/scopa

import time
#optimal_area =
min_area = 500

def transform_img(img, angle = 0, scale = 1, tr = (0,0)):

    '''Function rotating, rescaling and translating the card inside an image.
    No care about border points is taken (added points are black and points going
    outside the image are cutted).

    :param img: image to be rotated.
    :type img: numpy.ndarray

    :param angle: angle of rotation in degrees. Default to 0.
    :type angle: float

    :param scale: isotropic scale factor. Default to 1.
    :type scale: float

    :param tr: tuple containing the components of the translation vector (in pixel unit)
    :type tr: tuple

    :return: rotated image with same shape of the input one.
    :rtype: numpy.ndarray
    '''

    img_center = (int(img.shape[0]/2), int(img.shape[1]/2))

    trasl_mat = np.zeros((2,3))
    trasl_mat[0,2]=trasl_mat[0,2]+tr[0]
    trasl_mat[1,2]=trasl_mat[1,2]+tr[1]
    trasl_mat[0,0] = trasl_mat[1][1] = 1
    img = cv2.warpAffine(img, trasl_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
        # applies the transformation to each pixel:
        #       𝚍𝚜𝚝(x,y)=𝚜𝚛𝚌(𝙼00x+𝙼01y+𝙼02,𝙼10x+𝙼11y+𝙼12) with M = rot_mat
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
        # return a 2x3 matrix realizing rotation and rescaling
    img = cv2.warpAffine(img, rot_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
    return img


def transform_img_args(img, min_area):

    '''Function that, given an image with a unique card, returns a tuple with the
       argument required by the function transform_img in order to get an image with
       the card in the center and vertically orientated.

       :param img: input image
       :type img: numpy.ndarray

       :param min_area: minimum contour-area (in pixels) recognised as a (possible) card.
       :type img: int

       :return: tuple (img, angle, scale, tr)
       :rtype: tuple
    '''

    edges_img = cv2.Canny(img,100,200)
    contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours]
    contours = [contours[i] for i in range(len(areas)) if areas[i] > min_area]
    if len(contours)!=1:
        print(areas)
        print(f'Error: founded {len(contours)} contours instead of 1')
        cv2.drawContours(img, contours, -1, (0,0,255), 2)
        show_image(img)
    contour= contours[0]
    rect = cv2.minAreaRect(contour)
        # rect = ((x_c, y_c), (w,h), angle)
    box = np.int0(cv2.boxPoints(rect))

    print(f'card dimensions: {rect[1][1]} x {rect[1][0]}')

    #angle
    if rect[1][0]<rect[1][1]:
        angle = rect[2]
    else:
        angle = rect[2]+90
    #scale
    #scale = optimal_area/cv2.contourArea(contour)
    scale = 1
    print(cv2.contourArea(contour))
    #tr
    H, W = img.shape[0:2]
    tr = (int(W/2 - rect[0][0]), int(H/2 - rect[0][1]))

    return img, angle, scale, tr


def n_card_to_string(numb):
    if numb<11:
        return n_card_to_number(numb%10)+'_harts(cuori)'
    elif numb<21:
        return n_card_to_number(numb%10)+'_diamonds(quadri)'
    elif numb<31:
        return n_card_to_number(numb%10)+'_clubs(fiori)'
    elif numb<41:
        return n_card_to_number(numb%10)+'_spades(picche)'
    else:
        sys.exit(f'Error: invalid numb ({numb}) passed to n_card_to_string')

def n_card_to_number(numb):
    if numb%10 == 1: return 'A'
    if numb%10 == 8: return 'J'
    elif numb%10 == 9: return 'Q'
    elif numb%10 == 0: return 'K'
    else: return str(numb)


'''
test:
    - presa l'immagine img/pic_0.jpg portalaal centro
    - verifica che se ci fai transform_img_arg i results vengono circa zero
    - verifica che se la sposti e giri un po' i parametri vengono l'opposto di quello che hai fatto (circa)
'''

url = 'http://192.168.1.5:8080'

def import_deck(dir, start = 1):
    print('altezza circa 45 cm')
    i = start
    while i<41:
        print(i)
        print(f'Insert {n_card_to_string(i)}')
        img = import_img(url, True)
        img_array = get_cards(img, 500, verbouse=False)
        if len(img_array)!=1:
            print(f'Error: founded {len(img_array)} objects instead of 1')
            for i in range(0,len(img_array)):
                msk = img_array[i][:,:,0] + img_array[i][:,:,1] + img_array[i][:,:,2]
                img[msk!=0] = (0,0,255)
            show_image(img)
            # no parenthesis!
        img = transform_img(*transform_img_args(img_array[0], min_area))

        show_image(img, f'{n_card_to_string(i)}', 1000)
        inp = input('Save image? (y,n) [y]: ')
        if inp == 'y' or inp == '':
            cv2.imwrite(dir+f'/{n_card_to_string(i)}.jpg', img)
            i = i+1
        elif inp != 'n':
            print(f'Invalid input: {inp}')

import_deck('deck',41)

'''TO DO:
    - credo sia interesssante ridurre le immagini a erray (720,720,3) di 0 e 1 dividendo per 255 e arrotondando (anche meno di 720)
    - test e documentation
'''

    # some image hadling

for i in range (41,41):
    img = cv2.imread(f'deck0/{n_card_to_string(i)}.jpg')
    img = (img/255-0.15).round().astype(np.uint8)
    img = img*255
    cv2.imwrite(f'deck0m/{n_card_to_string(i)}.jpg', img)

for i in range (41,41):
    img = cv2.imread(f'deck0/{n_card_to_string(i)}.jpg')
    img1 = img[:,:,2] #only red
    #img = (img/255-0.15).round().astype(np.uint8)
    #img = img*255
    cv2.imwrite(f'deck0m1/{n_card_to_string(i)}.jpg', img1)


def generate_card(layers=1):
    card = random.randrange(1,41)
    angle=random.uniform(0,360)
    scale=random.uniform(0.9,1.1)

    if layers == 1:
        print('deck0m1/'+n_card_to_string(card)+'.jpg')
        img = cv2.imread('deck0m1/'+n_card_to_string(card)+'.jpg')
    elif layers == 3:
        print('deck0/'+n_card_to_string(card)+'.jpg')
        img = cv2.imread('deck0/'+n_card_to_string(card)+'.jpg')
    image_shape = img.shape
    marg = 5
    tr=(random.randrange(-int(image_shape[0]/2-image_shape[0]/marg), int(image_shape[0]/2-image_shape[0]/marg)),
        random.randrange(-int(image_shape[1]/2-image_shape[1]/marg), int(image_shape[1]/2-image_shape[1]/marg)))

    img = transform_img(img, angle, scale, tr)
    show_image(img, 'generated card', 100)
    return img

'''
inp = 3
while inp == 1 or inp == 3:
    generate_card(inp)
    inp = int(input('inp: '))
'''
