import sys
import numpy as np
import cv2
import urllib.request
import re
import logging
from tensorflow.keras.models import load_model, Model
from itertools import compress, product

logging.getLogger('tensorflow').disabled = True


def show_image(img, window_name = 'image', wait_key = 0):

    '''Function that simply shows the input image; press 'q' to close the window and proceed.

    :param img: input image.
    :type img: numpy.ndarray

    :param window_name: name of the window. Default 'image'
    :type window_name: string

    :param wait_key: delay in milliseconds. 0 is a special value meaning "forever"; Default 0.
    :type wait_key: int

    :return: 1 if "q" is pressed, 0 otherwise.
    :rtype: int
    '''

    cv2.startWindowThread()
    cv2.imshow(window_name,img)
    q = cv2.waitKey(wait_key) & 0xFF
    if q == ord("q") & 0xff:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return 1
    return 0
    '''se schiaccio un'altra lettera non esegue l'if, ma comunque esce dal wait waitKey
    accetto che l'aggiornamento del video non sia immediato'''


def import_img(url, show_video = False, save_img = None):

    '''Function capturing a colored image from IPcamera given the url.

    :param url: web address (Uniform Resource Locator) of the desidered IPcamera
    :type url: string

    :param show_video: if False: no captured image is shown and the image is taken immediately;
                       if True: the capturing video is shown and the image is taken when the
                       key "q" is pressed.
                       Default to False.
    :type show_video: Bool

    :param save_img: if different from None save the captured image with the
                     specified string plus the extension ".jpg".
                     Default to None.
    :type save_img: string

    :return: img as numpy 3d-array
    :rtype: numpy.ndarray
    '''

    url = url + '/shot.jpg'
    try:
        if show_video:
            while True:
                imgResp = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgNp, -1)
                stop = show_image(img,'IPWebcam', 5)
                if stop:
                    break
        else:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
    except urllib.error.URLError as urlerr:
        print(urlerr)
        sys.exit('Error: camera not correctly connected')

    if save_img is not None:
        cv2.imwrite(save_img+'.jpg',img)

    return img


def get_area(url):

    '''Function returning the area of the framed card.
    Only one card must be framed and the card must have
    an area superior to 5000 (pizels)

    :param url: web address (Uniform Resource Locator) of the desidered IPcamera
    :type url: string

    :return: area of the card
    :rtype: float
    '''

    img = import_img(url, True)
    filtered_img = cv2.bilateralFilter(img, 40,80,80)
    edges_img = cv2.Canny(filtered_img,120,200)
    contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours if cv2.contourArea(cont)> 5000]
    if len(areas)!=1:
        sys.exit(f'More than one object has been detected in get_area with areas = {areas}')
    else:
        return areas[0]


def transform_img(img, angle = 0, scale = 1, tr = (0,0), white = [0,0,0]):

    '''Function rotating, rescaling and translating the card inside an image.
    No care about border points is taken (added points are black and points going
    outside the image are cutted).

    :param img: image to be rotated.
    :type img: numpy.ndarray

    :param angle: angle of rotation in degrees counterclockwise. Default to 0.
    :type angle: float

    :param scale: isotropic scale factor. Default to 1.
    :type scale: float

    :param tr: tuple containing the components of the translation vector (in pixel unit)
    :type tr: tuple

    :return: transformed image with same shape of the input one.
    :rtype: numpy.ndarray
    '''

    img_center = (int(img.shape[0]/2), int(img.shape[1]/2))

    trasl_mat = np.zeros((2,3))
    trasl_mat[0,2]=tr[0]
    trasl_mat[1,2]=tr[1]
    trasl_mat[0,0] = trasl_mat[1][1] = 1
    img = cv2.warpAffine(img, trasl_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
        # applies the transformation to each pixel:
        #       𝚍𝚜𝚝(x,y)=𝚜𝚛𝚌(𝙼00x+𝙼01y+𝙼02,𝙼10x+𝙼11y+𝙼12) with M = rot_mat
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
        # return a 2x3 matrix realizing rotation and rescaling
    img = cv2.warpAffine(img, rot_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
    return img


def handle_border_problems(c_x,c_y):
    if c_x-128<0:       dx = 128-c_x+1
    elif c_x+128>720:  dx = -(128-720+c_x)-1
    else:               dx = 0
    if c_y -128<0:      dy = 128-c_y+1
    elif c_y+128>720:   dy = -(128-720+c_y)-1
    else:               dy = 0
    return int(dx),int(dy)

def get_cards(img, min_area, verbouse=False, margin=25, importing_deck = False):

    '''Function that given a 720x720x3 shaped image with some non-overlapping cards,
    returns an array of 256x256 images each containing one card in a thresholded style.

    :param img: image from which extract cards.
    :type img: numpy.ndarray

    :param min_area: minimum contour-area (in pixels) recognised as a card.
    :type img: float

    :param verbouse: if True shows the main steps done to get the result
                     (always press q to close the windows). Default to False.
    :type verbouse: bool

    :param margin: parameter regulating the margin in the upper_bound threshold in BG (see code and
                   note below).
                   Default to 25.
    :type margin: int

    :param importing_deck: If True returns a tuple with an image and the angle
                           required by the function transform_img in order to get an image with
                           the card in the center and vertically orientated. Default to False.
    :type importing_deck: bool

    :return: array of images
    :rtype: numpy.ndarray

    note: in order to get the cards the following steps are performed:

        * a filtered image is created in order to reduce the noise;
        * contours are founded using the `Canny <https://en.wikipedia.org/wiki/Canny_edge_detector>`_
          algorithm (see code for details);
        * only contours with area > min_area are kept;
        * images containing only one card are created using masks;
        * images are thresholded using the function cv2.inRange(); the lower_bound is fixed to [1,1,1],
          the upper_bound for Blue and Green is chosen basing on the white part of each card and on the
          parameter margin, for Red is fixed to 255 (max possible value);
        * images are properly cutted.
    '''

    filtered_img = np.zeros_like(img)
    filtered_img = cv2.bilateralFilter(img, 40,80,80)
        # the three parameters refer to:
        #       d:  Diameter of each pixel neighborhood that is used during filtering.
        #       sigmaSpace:	Filter sigma in the coordinate space
    edges_img = cv2.Canny(filtered_img,120,200)
        # the two parameters refer to:
        #       largest value:  used to find initial segments of strong edges
        #       smallest value: lower bound for weak edges
    contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.RETR_EXTERNAL:    retrieves only the extreme outer contours.
        #                       It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours
        # cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
        #                          and leaves only their end points
        # contours is a list whose elements are np.arrays containing the coordinates of border_points
        #                   its dimensions are len_contours x len_border_point x 1 x 2(x_y coordinates)
    areas = [cv2.contourArea(cont) for cont in contours]
    cards_contour = [contours[i] for i in range(len(areas)) if areas[i] > min_area]
    areas = [areas[i] for i in range(len(areas)) if areas[i] > min_area]

    cards_contour = [cont for x,cont in sorted(zip(areas,cards_contour))]
    # selection of the (possible) cards

    cards_img = []

    for i in range(len(cards_contour)):
        mask = np.zeros_like(img[:,:,0])
        cv2.drawContours(mask, cards_contour, i, 255, -1)
            # We create a mask where white is the card, black is everything else
            # note that argument -1 in drawContours generates filled contour in mask
        out = np.zeros_like(img)
        out[mask == 255] = filtered_img[mask == 255]
            # Use the mask to select the interesting pixels

        cv2.drawContours(out, cards_contour, i, (0,0,255), 3)
        rect = cv2.minAreaRect(cards_contour[i])    # rect=((x_c,y_c), (W, H), angle)
        w = int(rect[1][1]*np.sin(rect[2]*np.pi/180)/2.35)
        h = int(rect[1][1]*np.cos(rect[2]*np.pi/180)/2.35)
        x_conf = int(rect[0][0]) + w
        y_conf = int(rect[0][1]) - h
        out = cv2.inRange(out, np.array([1,1,1]), np.array([out[y_conf,x_conf,0]-margin, out[y_conf,x_conf,1]-margin, 255]))
        dx,dy = handle_border_problems(rect[0][0], rect[0][1])
        if (dx,dy) != (0,0):
            out = cv2.warpAffine(out, np.float32([[1,0,dx], [0,1,dy]]), out.shape[0:2])
        out = out[dy+int(rect[0][1])-128:dy+int(rect[0][1])+128,dx+int(rect[0][0])-128:dx+int(rect[0][0])+128]
        cards_img.append(out)

    if verbouse:
        show_image(img, 'Input image')
        show_image(filtered_img, 'Filtered image')
        show_image(edges_img, 'Edges image')
        cv2.drawContours(img, cards_contour, -1, (0,255,0), 2)
        show_image(img, 'Founded contours')
        n_cards = len(cards_img)
        print(f'Number of founded cards: {n_cards}')
        for i, imm in enumerate(cards_img):
            show_image(imm, f'Card {i} of {n_cards}')

    if importing_deck:
        if len(cards_img)!=1:
            print(f'Error: founded {len(cards_img)} objects instead of 1')
            for i in range(0,len(cards_img)):
                msk = cards_img[i][:,:,0] + cards_img[i][:,:,1] + cards_img[i][:,:,2]
                img[msk!=0] = (0,0,255)
            show_image(img)
        #angle
        if rect[1][0]<rect[1][1]:
            angle = rect[2]
        else:
            angle = rect[2]+90

        return out, angle

    return np.array(cards_img)


def convert_card(input):

    '''Function converting the input between the three different formats:

        * string 'n_suit' to tuple (n, suit)
        * int 1->40 to tuple (n, suit)
        * tuple (n, suit) to string 'n_suit'
    '''

    if type(input) == str:
        suit_d = { 'hearts':0, 'diamonds':1, 'clubs':2, 'spades':3 }
        splitted = input.split('_')
        return (int(splitted[0]), suit_d[splitted[1]])
    if type(input) == int:
        n = input%10 if input%10!=0 else 10
        #return convert_card( (n, int((input-1)/10)+1) )
        return (n, int((input-1)/10))
    elif type(input) == tuple:
        suit_d = {0:'hearts', 1:'diamonds', 2:'clubs', 3:'spades' }
        return f'{input[0]}_{suit_d[input[1]]}'


def string_to_n_card(str):

    '''Function returning the number (1->40) related to a card.
    This is used in generate_cards to order the cards.

    :param str: string describing the card; the format can be 'n_suit' or a
                path like '.path/n_suit.jpg'
    :type str: string
    '''

    if str.startswith('.'):
         words = re.split('_|/|.j', str)[-3:-1]
    else: words = str.split('_')
    suit_to_n = {'hearts':0, 'diamonds':10, 'clubs':20, 'spades':30}
    return int(words[0])+suit_to_n[words[1]]


def generate_cards(n=1, layers=3, ang = True, traslate=False):

    '''Function generating an array of images each conteining a random card in a random position
    starting from the images in deck_thr and deck0 directories.

    :param layers: specifies the type of the image generated (colored if layers = 3,
                   black&white corresponding to the threshlded image if layers = 1).
                   Default to 3.
    :type layers: int

    :param ang: specifies if images generated have to be randomly orientated. Default to True.
    :type ang: bool

    :param traslate: specifies if images generated have to be randomly traslated. Default to False.
    :type traslate: bool

    :return: rotated image with same shape of the input one.
    :rtype: numpy.ndarray
    '''

    cards = np.random.randint(1,41,n, np.uint8)
    if ang:
        angles= np.random.uniform(0,360,n)
    else:
        angles = np.full(n, 0.)
    scales= np.random.uniform(0.9,1.1,n)
    if layers == 1:
        inp_images = sorted([file for file in glob.glob("./deck_thr/*.jpg")], key = string_to_n_card)
        inp_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in inp_images]
    elif layers == 3:
        inp_images = sorted([file for file in glob.glob("./deck0/*.jpg")], key = string_to_n_card)
        inp_images = [cv2.imread(file)[232:488,232:488] for file in inp_images]
    image_shape = inp_images[0].shape

    if traslate:
        marg = 5
        max_tr_x = int(image_shape[0]/2-image_shape[0]/marg)
        max_tr_y = int(image_shape[1]/2-image_shape[1]/marg)
        tr_x = np.random.randint(0,max_tr_x,n)
        tr_y = np.random.randint(0,max_tr_y,n)
    else:
        tr_x = np.zeros(n)
        tr_y = np.zeros(n)

    images = [transform_img(inp_images[cards[i]-1], angles[i], scales[i], (tr_x[i], tr_y[i]))
              for i in range(len(cards))]

    return np.array(images), np.array(cards)


def transform_img(img, angle = 0, scale = 1, tr = (0,0)):

    '''Function rotating, rescaling and translating the card inside an image.
    No care about border points is taken (added points are black and points going
    outside the image are cutted).

    :param img: image to be rotated.
    :type img: numpy.ndarray

    :param angle: angle of rotation in degrees counterclockwise. Default to 0.
    :type angle: float

    :param scale: isotropic scale factor. Default to 1.
    :type scale: float

    :param tr: tuple containing the components of the translation vector (in pixel unit)
    :type tr: tuple

    :return: transformed image with same shape of the input one.
    :rtype: numpy.ndarray
    '''

    img_center = (int(img.shape[0]/2), int(img.shape[1]/2))

    trasl_mat = np.zeros((2,3))
    trasl_mat[0,2]=tr[0]
    trasl_mat[1,2]=tr[1]
    trasl_mat[0,0] = trasl_mat[1][1] = 1
    img = cv2.warpAffine(img, trasl_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
        # applies the transformation to each pixel:
        #       𝚍𝚜𝚝(x,y)=𝚜𝚛𝚌(𝙼00x+𝙼01y+𝙼02,𝙼10x+𝙼11y+𝙼12) with M = rot_mat
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
        # return a 2x3 matrix realizing rotation and rescaling
    img = cv2.warpAffine(img, rot_mat, img.shape[0:2], flags=cv2.INTER_LINEAR)
    return img


def import_deck(path, url, start = 1):

    '''Function importing the images of the cards in the deck.

    :param path: path to the directory where the deck is going to be saved(the directory
                 must be previously created).
    :type path: string

    :param url: web address (Uniform Resource Locator) of the desidered IPcamera.
    :type url: string

    :param start: number of the starting acquisition card. Default to 1 (corresponding to
                  A_harts)
    :type start: int
    '''

    print('altezza circa 40 cm')
    i = start
    while i<41:
        print(i)
        print(f'Insert: '+ convert_card(convert_card(i)))
        img = import_img(url, True)
        card_img, ang = get_cards(img, 5000, verbouse=False, importing_deck=True)
        card_img = transform_img(card_img, angle = ang)
        show_image(card_img, convert_card(convert_card(i)), 1000)
        inp = input('Save image? (y,n) [y]: ')
        if inp == 'y' or inp == '':
            print(path+'/'+convert_card(convert_card(i))+'.jpg')
            cv2.imwrite(path+'/'+convert_card(convert_card(i))+'.jpg', card_img)
            i = i+1
        elif inp != 'n':
            print(f'Invalid input: {inp}')


def met1_to_cards(cards, numb_model, suit_model):
    pred_n = numb_model.predict(cards)
    pred_s = suit_model.predict(cards)
    return [np.argmax(pred_n[i])+1 + 10*np.argmax(pred_s[i]) for i in range(len(cards))]

def met2_to_cards(cards, all_model_1):
    pred_1 = all_model_1.predict(cards)
    return [np.argmax(pred_1[i])+1 for i in range(len(cards))]

def met3_to_cards(cards, all_model_2):
    [pred_2_n, pred2_s] = all_model_2.predict(cards)
    return [np.argmax(pred_2_n[i])+1 + 10*np.argmax(pred2_s[i]) for i in range(len(cards))]


def image_to_cards(img, min_area, margin, path_to_models = './models'):

    '''Function using models to detect cards from image
    '''

    numb_model = load_model(path_to_models+'/numb_model.h5')             # model_number.ipynb
    suit_model = load_model(path_to_models+'/suit_model.h5')             # model_suit.ipynb
    all_model_1 = load_model(path_to_models+'/all_model_1.h5')           # card-rec1layer_alltogether.ipynb
        # these model returns an array of lenght n, with predictions in one hot notation
    all_model_2 = load_model('models/all_model_2.h5')           # card-rec1layer_alltogether2.ipynb
        # this model return a list with the 2 predictions in one hot notation

    cards = get_cards(img, min_area, False, margin)
    cards_rot = get_cards(transform_img(img, angle = 90), min_area, False, margin)
    assert(len(cards) == len(cards_rot))
    n_cards = len(cards)

    cards_original = np.array([met1_to_cards(cards, numb_model, suit_model), met2_to_cards(cards, all_model_1), met3_to_cards(cards, all_model_2)]).T
    cards_rotated = np.array([met1_to_cards(cards_rot, numb_model, suit_model), met2_to_cards(cards_rot, all_model_1), met3_to_cards(cards_rot, all_model_2)]).T

    c = [np.concatenate([cards_original[i],cards_rotated[i]]) for i in range(n_cards)]
    '''
    sure_cards = []
    for i in range (n_cards):
        freq_i = np.bincount(c[i])
        if max(freq_i) > 4:
            found_card = convert_card(int(np.argmax(freq_i)))

        else:
            print(f"The outcome isn't sure about one of the cards")
            show_image(cards[i], 'uncertain card')
            found_card = convert_card(input('Please, insert the uncertain card in format n_suit: '))
            # prima chiudi l'immagine con q...
        '''
    sure_cards = []
    for i in range (n_cards):
        freq_i = np.bincount(c[i])
        found_card = convert_card(int(np.argmax(freq_i)))
        # if there are equal frequences the first one is returned.
        print(convert_card(found_card))
        sure_cards.append(found_card)

    inp = input('Have the cards been correctly recognised? (y,n) [y]')
    if not( inp == '' or inp == 'y' ):
        for i, card in enumerate (sure_cards):
            print(i, convert_card(card))
        while True:
            str = input("Insert wrong card in format 'index to n_suit' (press Enter if everything is ok): ")
            if str == '': break
            try:
                [index, _ , correct_card] = str.split()
            except ValueError:
                print('Wrong sintax: the correct format is index to n_suit')
                continue
            index = int(index)
            print(correct_card)
            sure_cards[index] = convert_card(correct_card)
            print(sure_cards[index])

    return sure_cards


class Deck:

    '''Class handling operations concerning a group of cards.
    The unique attribute is a list containg the cards in the format (n, suit) and is
    treated as private. Setter and getter are implemented as indicated in Match documentation
    (see code for details).

    The methods are:

    * number_cards
    * get_possibilities

    '''

    def __init__(self, list=[]):
        self._cards = list

    @property
    def cards(self):
        '''_cards getter'''
        return self._cards

    @cards.setter
    def cards(self, tup):
        '''_cards handler: tup is a tuple (mode, list)

            * if mode == 1 adds elements in list to _cards
            * if mode == 0 removes elements in list from _cards
        '''

        (mode, list) = tup
        if mode==0:
            for i in range(len(list)):
                self._cards.remove(list[i])

        elif mode==1:
            self._cards = self._cards + list

    def number_cards(self, n_s):

        '''Returns the number of cards in _cards:

            * of a given suit if n_s is one of {'h', 'd', 'c', 's'}
            * of a given number if n_s is a number between 1 and 10
        '''

        n=0
        if type(n_s)==str:
            str_to_suit = {'h':0, 'd':1, 'c':2, 's':3}
            for card in self._cards:
                if card[1] == str_to_suit[n_s]: n=n+1
        elif type(n_s)==int:
            for card in self._cards:
                if card[0] == n_s: n=n+1
        return n

    def get_possibilities(self, num):

        '''Method returning the possible combinations of cards that can be taken
        if a card whose number is num is played.
        '''

        if num == 1:
            ace_in_table = np.array([(1,0) in self._cards, (1,1) in self._cards, (1,2) in self._cards, (1,3) in self._cards])
            if ace_in_table.sum() == 0:
                return [self._cards[:]]
            else:
                aces_list = [(1,0), (1,1), (1,2), (1,3)]
                return [ [aces_list[i]] for i in range(4) if ace_in_table[i]]  # verifica questo!!!!
        comb = list((list(compress(self._cards,mask)) for mask in product(*[[0,1]]*len(self._cards))))
        good_comb = []
        for i in range(0, len(comb)):
            sum = 0
            for j in range(0, len(comb[i])):
                sum = sum + comb[i][j][0]
            if  sum == num:
                good_comb.append(comb[i])
        return good_comb


class Match:

    '''Class handling the match cpu VS other.
    The attributes (handled as private), are (with self_explaining notation):

    * _not_played_deck
    * _cpu_deck
    * _other_deck
    * _table
    * _cpu_hand
    * _other_score
    * _cpu_score
    * _cpu_last_take: 1 if the last take has been peformed by the cpu

    The first 5 attributes are Deck objects, the other are integers.
    For each of the first 5 attributes a getter and a setter are implemented using decorators.
    For the scores only the getter is implemented.
    The setters for the Deck objects are implemented in an unconventional way (see code
    for details); substantially:

    * if mode == 1: list passed is added to the Deck object;
    * if mode == 0: list passed is removed from the Deck object;

    The methods of the Class are:

    * card_value
    * cpu_plays
    * play_card_outcome
    * other_plays
    * mano
    * end_game

    Moreover the special methods __call__ and __add__ are implemented.
    '''

    def __init__(self, cpu_score = 0, other_score = 0):
        self._not_played_deck = Deck([(i,j) for j in range (0,4) for i in range(1,11)])
        self._cpu_deck = Deck([])
        self._other_deck = Deck([])
        self._table = Deck([])
        self._cpu_hand = Deck([])
        self._other_score = other_score
        self._cpu_score = cpu_score
        self._cpu_last_take = 0


    @property
    def not_played_deck(self):
        '''_not_played_deck getter'''
        return self._not_played_deck

    @not_played_deck.setter
    def not_played_deck(self, tup):
        '''_not_played_deck handler, see Deck.cards setter for details'''
        self._not_played_deck.cards = tup

    @property
    def cpu_deck(self):
        '''_cpu_deck getter'''
        return self._cpu_deck

    @cpu_deck.setter
    def cpu_deck(self, tup):
        '''_cpu_deck handler, see Deck.cards setter for details'''
        self._cpu_deck.cards = tup

    @property
    def other_deck(self):
        '''_other_deck getter'''
        return self._other_deck

    @other_deck.setter
    def other_deck(self, tup):
        '''_other_deck handler, see Deck.cards setter for details'''
        self._other_deck.cards = tup

    @property
    def table(self):
        '''_table getter'''
        return self._table

    @table.setter
    def table(self, tup):
        '''_table handler, see Deck.cards setter for details'''
        self._table.cards = tup

    @property
    def cpu_hand(self):
        '''_cpu_hand getter'''
        return self._cpu_hand

    @cpu_hand.setter
    def cpu_hand(self, tup):
        '''_cpu_hand handler, see Deck.cards setter for details'''
        self._cpu_hand.cards = tup

    @property
    def cpu_score(self):
        '''_cpu_score getter'''
        return self._cpu_score

    @property
    def other_score(self):
        '''_other_score getter'''
        return self._other_score

    def card_value(self, card, in_hand = False):

        '''Method returning the value of a card given the actual state of the game.
        The values over which the algorithm choose the best play are defined inside this function
        and can be modified to optimize and personalize the strategy of the game.

        :param card: card of which the value must be calculated; the accepted format is (n, suit)
        :type card: tuple

        :param in_hand: the argument is used only when calculating the value of an ace
        :type in_hand: bool

        :return: value of the passed card
        :rtype: int
        '''

        single_c = 0.4 if ( len(self._cpu_deck.cards) < 21 and len(self._other_deck.cards) < 21 ) else 0.
        value_c = 0.02
        diamond_c = 0.3 if (self._cpu_deck.number_cards('d')<6 and self._other_deck.number_cards('d')<6) else 0.
        seven_c = 0.4 if (self._cpu_deck.number_cards('d')<3 and self._other_deck.number_cards('d')<3) else 0.
        seven_c2 = 0.1 if (7,1) in self._not_played_deck.cards else 0.
        sevenD_c = 0.2
        ace_c = -0.5
        scopa = -0.2

        val = single_c + card[0]*value_c
        if card[1]==1: val = val + diamond_c
        if card[0] == 7:
            val = val + seven_c
            if card[1] == 1: val = val + sevenD_c
        if in_hand and card[0]==1: val = val + ace_c

        sum_in_table = np.array([item[0] for item in self._table.cards]).sum() - card[0]
        if (sum_in_table < 11 and sum_in_table > 0 and in_hand):
            val  = val + scopa

        return val

    def cpu_plays(self):

        '''Method handling the operations performing a cpu_play:
        The best play is chosen using the methods play_card_outcome and card_value.
        '''

        best_card = self._cpu_hand.cards[0]
        best_val, best_take = self.play_card_outcome(self._cpu_hand.cards[0])
        for i in range (1, len(self._cpu_hand.cards)):
            val, take = self.play_card_outcome(self._cpu_hand.cards[i])
            if val > best_val: best_card, best_val, best_take = self._cpu_hand.cards[i], val, take
        # interface
        if len(best_take)==0:
            str_g = 'nothing'
        else:
            self._cpu_last_take = 1
            str_g = ''
            for card in best_take:
                str_g = str_g+' '+convert_card(card)
        print('cpu plays: '+convert_card(best_card)+'\ttaking: '+str_g)
        #decks handling
        self._cpu_hand.cards = (0, [best_card])
        self._not_played_deck.cards = (0, [best_card])
        if len(best_take)==0:
            if best_card[0]!=1: self._table.cards = (1, [best_card])
            else:               self._cpu_deck.cards = (1, [best_card])
        else:
            self.table.cards = (0, best_take)
            self._cpu_deck.cards = (1, [best_card]+best_take)

        if (len(self._table.cards) == 0 and best_card[0] != 1):
            print('cpu made scopa')
            self._cpu_score = self._cpu_score +1

        print('--- end of cpu turn ---')
        print(f'cards in table: {self._table.cards}')
        print(f'cards in cpu_deck: {self._cpu_deck.cards}')

    def play_card_outcome(self, card):

        '''Method returning the outcome of the best play that can be done given
        the actual state of the match and the passed card.
        The best play is calculated basing on the card_value method.

        :return: tuple containg the value of the best possibility and a list containg
                 the cards that can be taken playing that card.
        :rtype: tuple
        '''

        possible_plays = self._table.get_possibilities(card[0])
        if len(possible_plays)==0:
            best_val = - self.card_value(card, in_hand = True)
            best_take = []
        else:
            best_val = -1e3
            for l in possible_plays:
                val  = self.card_value(card, in_hand = True)
                for c in l:
                    val = val + self.card_value(c)
                if val>best_val:
                    best_val = val
                    best_take = l
        return best_val, best_take

    def other_plays(self):

        '''Method handling the operations performing an other_play:
        The method asks for the card played by the other player and this must be
        passed with notation n_suit (es: '4_hearts').
        If there is more than one possibility, a question is raised by the algorithm.
        Few modifications to the code are required to use the camera also to recognise
        the cards played by other (but in order to make this convenient a second camera
        should be used).
        '''

        try:
            card = input('Other plays the card (format n_suit): ')
            card = convert_card(card)
        except KeyError:
            card = input('Not valid input; try again (format n_suit): ')
            card = convert_card(card)
        possible_plays = self._table.get_possibilities(card[0])
        if len(possible_plays) == 0:
            print('Other takes nothing')
            self._table.cards = (1, [card])
        elif len(possible_plays) == 1:
            self._cpu_last_take = 0
            self._table.cards = (0, possible_plays[0])
            self._other_deck.cards = (1, possible_plays[0]+[card])
            str_g = ''
            for c in possible_plays[0]:
                str_g = str_g+' '+convert_card(c)
            print(f'Other takes: ' + str_g)
        else:
            self._cpu_last_take = 0
            for i in range(len(possible_plays)):
                str_g = ''
                for c in possible_plays[i]:
                    str_g = str_g+' '+convert_card(c)
                print(f'{i}\t'+ str_g)
            inp = int(input(f'Other takes (type index): '))
            self._table.cards = (0, possible_plays[inp])
            self._other_deck.cards = (1, possible_plays[inp]+[card])
        self._not_played_deck.cards = (0, [card])

        if (len(self._table.cards) == 0 and card[0] != 1):
            print('other made scopa')
            self._other_score = self._cpu_score +1

        print('--- end of other turn ---')
        print(f'cards in table: {self._table.cards}')
        print(f'cards in other_deck: {self._other_deck.cards}')

    def mano(self, url, min_area, margin, path_to_models):

        '''Method handling the operations performing a set of three plays.
        '''

        print('Show cards in cpu_hand')
        cpu_img = import_img(url, show_video = True)
        cpu_cards = image_to_cards(cpu_img, min_area, margin, path_to_models)
        self._cpu_hand.cards = (1, cpu_cards)
        for i in range(3):
            self.cpu_plays()
            self.other_plays()

    def end_match(self, url, min_area, margin, path_to_models):

        '''Method calculating scores at the end of the match.
        '''

        if self._cpu_last_take == 1:
            self._cpu_deck.cards = (1, self._table.cards)
        else:
            self._other_deck.cards = (1, self._table.cards)

        partial_cpu = []
        partial_other = []
        # cards
        if len(self._cpu_deck.cards)>20: partial_cpu.append('cards')
        elif len(self._other_deck.cards)>20: partial_other.append('cards')

        # diamonds
        if self._cpu_deck.number_cards('d')>5: partial_cpu.append('diamonds')
        elif self._other_deck.number_cards('d')>5: partial_other.append('diamonds')

        # 7_diamonds
        if (7,1) in self._cpu_deck.cards: partial_cpu.append('7_d')
        else: partial_other.append('7_d')

        # primero (non so le regole esatte...)
        order_of_importance = [7,6,1]
        for i in order_of_importance:
            if self._cpu_deck.number_cards(i) > 2:
                partial_cpu.append('primero')
                break
            elif self._other_deck.number_cards(i) > 2:
                partial_other.append('primero')
                break

        self._cpu_score = self._cpu_score + len(partial_cpu)
        self._other_score = self._other_score + len(partial_other)

        print(f'partial_cpu: \t {len(partial_cpu)} {partial_cpu}')
        print(f'partial_other: \t {len(partial_other)} {partial_other}')
        print(f'\tTot_cpu:\t{self._cpu_score}')
        print(f'\tTot_other:\t{self._other_score}')

        inp = input('Do you want to continue? (y,n) [y]: ')
        if (inp == 'y' or inp ==''):
            new_mat = Match(self._cpu_score, self._other_score)
            new_mat(url, min_area, margin, path_to_models)
        elif inp == 'n':
            if self._cpu_score == self._other_score:
                print('Match ended with tie')
            else:
                winner = 'cpu' if self._cpu_score > self._other_score else 'other'
                print('The winner is '+winner)

    def __call__(self, url, min_area, margin=25, path_to_models = './models'):

        '''Special method starting a new match
        '''

        print('Start of the match\nDeal the cards and show cards in table')
        table_img = import_img(url, show_video = True)
        table_cards = image_to_cards(table_img, min_area, margin, path_to_models)
        self._table.cards = (1, table_cards)
        self._not_played_deck.cards = (0, table_cards)

        t=0
        while len(self._not_played_deck.cards) > 0:
            print(f'\t*** Turn {t} *** {len(self._not_played_deck.cards)}')
            self.mano(url, min_area, margin, path_to_models)
            t=t+1

        self.end_match(url, min_area, margin, path_to_models)

    def __add__(self, other_match):
        return Match(self._cpu_score+other_match.cpu_score, self._other_score+other_match.other_score)
