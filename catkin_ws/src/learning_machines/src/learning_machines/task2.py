import numpy as np
from matplotlib import pyplot as plt
from data_files import FIGRURES_DIR
import cv2
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


def get_image(robobo: IRobobo):
    img = robobo.get_image_front()
    # encode to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv


def apply_mask(img):
    # TODO: test if this is good for both sim and irl
    return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def set_resolution(img):
    # TODO: check irl resolution of camera
    # sim photo format is (512, 512)
    return cv2.resize(img, (128, 128))


# other ideas:
# blurring/smoothing
# contour/edge detection

def process_image(img):
    return set_resolution(apply_mask(img))
