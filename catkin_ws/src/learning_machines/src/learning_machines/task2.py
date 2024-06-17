"""
NOTES FOR TASK 2
# Randomize the object generation (through lua)
# Randomize the starting point OR 1000 steps in one arena, then 1000 in another arena, etc etc

# Reality vs Simulation
Add noise to the simulated camera to help with reality adjustment
Randomize the camera orientation slightly

# Reward function:
The center of the object should be as close as possible.
-> what if there are two objects? steer to the closest one (largest area)

Multiplying the different objectives: robot pays equal attention to both
Adding the different objectives: robot can do one or the other independently

## What do we reward?
- Finding new greens
- The DIFFERENCE in green area, vs one frame and the next

# Image processing:
Use OpenCV
Normalize image dimensions!
Filter for green, use HSV
Try to find the center of the object


TASK FLOWCHART
- Search for objects
- Move to the closest object (the closest being the largest sized mask)
- Collide with it! (move forward until it disappears)
- If you do not see green, reward the robot for turning (UNTIL green is seen, then punish staying in place)
"""

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


def add_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.add(img, gaussian_noise)
    return noisy_image


# other ideas:
# blurring/smoothing
# contour/edge detection

def process_image(img):
    return set_resolution(apply_mask(img))


def process_image_w_noise(img):
    return set_resolution(apply_mask(add_noise(img)))
