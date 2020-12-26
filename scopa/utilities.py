"""Defines main utilities functions.
"""

import sys
import numpy as np
import cv2
import urllib.request


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

    cv2.imshow(window_name,img)
    q = cv2.waitKey(wait_key)
    if q == ord("q"):
        cv2.destroyAllWindows()
        return 1
    return 0


def import_img(url, show_video = False, save_img = None):

    '''Function capturing a colored image from IPcamera given the url.

    :param url: web address (Uniform Resource Locator) of the desidered IPcamera
    :type url: string

    :param show_video: if False: no captured image is shown and the image is taken immediatly;
                       if True: the capturing video is shown and the iamge is taken when the
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

    try:
        if show_video:
            while True:
                imgResp = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgNp, -1)
                stop = show_image(img,'IPWebcam', 3)
                if stop:
                    break
        else:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
    except urllib.error.URLError as urlerr:
        print(urlerr)
        inp = input('Error: camera not connected. Try again? (y/n) [y]: ')
        if inp == 'y' or inp == '':
            img = import_img(url, show_video, save_img)
            return img
        elif inp == 'n':
            sys.exit('Error: camera not connected')
        else:
            sys.exit('Error: invalid input')


    if save_img is not None:
        cv2.imwrite(save_img+'.jpg',img)

    return img


def get_cards(img, min_area, verbouse=False):

    '''Function that given an image returns an array of same shaped images each
    containing only one card.

    :param img: image from which extract cards.
    :type img: numpy.ndarray

    :param min_area: minimum of the area recognised as a (possible) card.
    :type img: int

    :param verbouse: if True shows the main steps done to get the result
                     (always press q to close the windows).Default to False.
    :type img: bool

    :return: array of images
    :rtype: numpy.ndarray

    note: in order to get the cards the following steps are performed:

        * a filtered image is created in order to reduce the noise;
        * contours are founded using the `Canny <https://en.wikipedia.org/wiki/Canny_edge_detector>`_
          algorithm (see code for details);
        * only contours with area > min_area are kept;
        * images containing only one card are created using masks.
    '''

    filtered_img = np.zeros_like(img)
    filtered_img = cv2.bilateralFilter(img, 40,80,80)
        # the three parameters refer to:
        #       d:  Diameter of each pixel neighborhood that is used during filtering.
        #       sigmaColor:	Filter sigma in the color space
        #       sigmaSpace:	Filter sigma in the coordinate space
    edges_img = cv2.Canny(filtered_img,100,200)
        # the two parameters refer to:
        #       largest value:  used to find initial segments of strong edges
        #       smallest value: lower bound for weak edges
    contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.RETR_EXTERNAL:    retrieves only the extreme outer contours.
        #                       It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours
        # cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
        #                          and leaves only their end points
        # contours is a list whose elements are np.arrays containing the coordinates of border_points
    areas = [cv2.contourArea(cont) for cont in contours]
    cards_contour = [contours[i] for i in range(len(areas)) if areas[i] > min_area]
        # selection of the (possible) cards

    cards_img = []

    for i in range(len(cards_contour)):
        mask = np.zeros_like(img[:,:,0])
        cv2.drawContours(mask, cards_contour, i, 255, -1)
            # We create a mask where white is what we want, black is everything else
            # note that argument -1 in drawContours generates filled contour in mask
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
            # Use the mask to select the interesting pixels
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

    return np.array(cards_img)






#img = import_img('http://192.168.1.5:8080/shot.jpg', 1)
#img = cv2.imread('img/get_cards0.jpg',cv2.IMREAD_COLOR)
#get_cards(img, 50, True)
