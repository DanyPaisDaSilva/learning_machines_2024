import cv2
import numpy as np


##################
# img operations #
##################

def apply_mask(img, state="RED"):
    if state == "RED":
        # apply red mask
        return apply_red_mask(img)
    else:
        # apply green mask
        return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def apply_red_mask(img):
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(img, lower_red, upper_red)
    mask2 = cv2.inRange(img, lower_red2, upper_red2)

    return mask1 + mask2


def set_resolution(img, resolution=64):
    # sim photo format is (512, 512)
    # irl photo format is (1536, 2048)
    return cv2.resize(img, (resolution, resolution))


def add_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.add(img, gaussian_noise)
    return noisy_image


def max_pooling(img):
    kernel_size = (8, 8)  # This is equivalent to a 2x2 pooling window in max pooling
    kernel = np.ones(kernel_size, np.uint8)

    # Perform dilation to simulate max pooling
    return cv2.dilate(img, kernel, iterations=1)


def process_image(img):
    return binarify_image(set_resolution(apply_morphology(max_pooling(apply_mask(img)))))


def binarify_image(img):
    _, binary_image = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return (binary_image / 255).astype(np.uint8)


def process_image_w_noise(img):
    return binarify_image(apply_morphology(set_resolution(apply_morphology(apply_mask(add_noise(img))))))


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
