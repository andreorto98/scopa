"""Defines main utilities functions.
"""

import sys
import numpy as np
import cv2
import urllib.request
import ssl

def ssquare(x):             # RICORDA DI ELIMINARLA...
    return x*x


def import_img(url, show_video = 0, save_img = None):

    '''Function capturing a colored image from IPcamera given the url.

    :param url: Web address (Uniform Resource Locator) of the desidered IPcamera
    :type url: string

    :param show_video: If 0: no captured image is shown and the image is taken immediatly;
                       If 1: the capturing video is shown and the iamge is taken when the
                       key "q" is pressed.
                       Default to 0.
    :type show_video: int

    :param save_img: If different from None save the captured image with the
                     specified string plus the extension ".jpg".
                     Default to None.
    :type save_img: string

    :return: img as numpy 3d-array
    :rtype: numpy.ndarray
    '''

    try:
        if show_video==0:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
        elif show_video==1:
            while True:
                imgResp = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                img = cv2.imdecode(imgNp, -1)
                cv2.imshow('IPWebcam',img)
                q = cv2.waitKey(10)  # wait 10 ms and then proceeds
                if q == ord("q"):
                    cv2.destroyAllWindows()
                    break
    except urllib.error.URLError as urlerr:
        print(urlerr)
        inp = input('Error: camera not connected. Try again? (y/n) [y]: ')
        if inp == 'y' or inp == '':
            img = import_img(url, show_video, save_img)
            return img
        elif inp == 'n':
            sys.exit("Error: camera not connected")

    if save_img is not None:
        cv2.imwrite(save_img+'.jpg',img)

    return img




#print(type(import_img('http://192.168.1.5:8080/shot.jpg', 0, 'try')))
