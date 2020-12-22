"""Defines the main functions importing images from the camera.
"""

import numpy as np
import cv2
import urllib.request
import ssl

def ssquare(x):
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
    :rtype: numpy nd-array

    '''

    url = 'http://192.168.1.5:8080/shot.jpg'

    img = cv2.imread(name_pic,cv2.IMREAD_COLOR)

    print('Take a picture')
    if img is None:
        while True:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            cv2.imshow('IPWebcam',img)
            q = cv2.waitKey(100)  # wait 10 ms and then proceeds
            if q == ord("q"):
                cv2.imwrite(name_pic,img)
                break;
            cv2.destroyAllWindows()
