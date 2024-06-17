import random

import pygame
import cv2
import numpy as np
import sys
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Set up the display
window_width = 1400
window_height = 900
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Image Processing Interface')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Define font
font = pygame.font.SysFont(None, 36)

# Button dimensions
button_width = 450
button_height = 50

# Button texts
button_texts = ['Resize Image', 'Add Noise', 'Color Mask', 'Apply Morphology', 'Draw Contours', 'Reset Image']

# Dummy variables
dummy_vars = ['Var 1: N/A', 'Var 2: N/A', 'Var 3: N/A']
def import_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def display_image(window, image):
    img_surface = pygame.surfarray.make_surface(image)
    img_rect = img_surface.get_rect(center=(450, 450))
    window.blit(img_surface, img_rect)

def draw_button(window, text, x, y):
    pygame.draw.rect(window, GRAY, (x, y, button_width, button_height))
    text_surf = font.render(text, True, BLACK)
    text_rect = text_surf.get_rect(center=(x + button_width // 2, y + button_height // 2))
    window.blit(text_surf, text_rect)

def set_resolution(img):
    # TODO: check irl resolution of camera
    # sim photo format is (512, 512)
    return cv2.resize(img, (128, 128))
def add_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.add(img, gaussian_noise)
    return noisy_image

# Bright green: 119, 96, 45
# Dark green: 120, 100, 33
def apply_mask(img):
    # TODO: test if this is good for both sim and irl
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (45, 70, 70), (85, 255, 225))
def apply_morphology(image):
    # Step 2: Closing operation to fill small holes
    closing_kernel_size = 5  # Kernel size for closing
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel)

    opening_kernel_size = 3  # Small kernel size for opening
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, opening_kernel)

    # Perform the 'opening' operation, which is equivalent to erosion followed by dilation.
    return opened_image
def find_contours(mask):
    """
    Find contours in the binary mask.

    Parameters:
    mask (np.array): Binary mask image.

    Returns:
    list: List of contours found in the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def calculate_blob_area(contour):
    """
    Calculate the area of a contour.

    Parameters:
    contour (np.array): Contour points.

    Returns:
    float: Area of the contour.
    """
    return cv2.contourArea(contour)

def get_largest_blob_area(contours):
    """
    Get the area of the largest blob (contour).

    Parameters:
    contours (list): List of contours.

    Returns:
    float: Area of the largest blob.
    """
    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    return calculate_blob_area(largest_contour)

def draw_contours(image, contours):
    """
    Draw each contour in a different color on the image.

    Parameters:
    image (np.array): Original image.
    contours (list): List of contours.

    Returns:
    np.array: Image with contours drawn in different colors.
    """
    # Make a copy of the image to draw contours on
    image_with_contours = image.copy()

    # Draw each contour with a random color
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(image_with_contours, [contour], -1, color, 2)  # Random color, thickness 2

    return image_with_contours
def reset_image(original_image):
    return original_image.copy()

def operation4(image):
    # Dummy operation: Edge Detection
    return cv2.Canny(image, 100, 200)

def display_dummy_vars(window, dummy_vars):
    y_offset = 800
    for var in dummy_vars:
        var_surf = font.render(var, True, BLACK)
        window.blit(var_surf, (950, y_offset))
        y_offset += 40


def main():
    clock = pygame.time.Clock()
    original_image = None
    image = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                x, y = event.pos
                if 950 <= x <= 950 + button_width:
                    if 50 <= y <= 100:
                        if image is not None:
                            image = set_resolution(image)
                    elif 150 <= y <= 200:
                        if image is not None:
                            image = add_noise(image)
                    elif 250 <= y <= 300:
                        if image is not None:
                            image = apply_mask(image)
                    elif 350 <= y <= 400:
                        if image is not None:
                            image = apply_morphology(image)
                    elif 450 <= y <= 500:
                        if image is not None:
                            image = draw_contours(image, find_contours(image))
                    elif 550 <= y <= 600:
                        if image is not None:
                            image = reset_image(original_image)
            elif event.type == KEYDOWN:
                if event.key == K_o:
                    file_path = input("Enter the image path: ")
                    image = import_image(file_path)
                    original_image = image.copy()

        window.fill(WHITE)

        # Draw image viewing space
        pygame.draw.rect(window, BLACK, (0, 0, 900, 900), 2)
        if image is not None:
            display_image(window, image)

        # Draw buttons
        for i, text in enumerate(button_texts):
            draw_button(window, text, 950, 50 + i * 100)

        # Display dummy variables
        display_dummy_vars(window, dummy_vars)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
