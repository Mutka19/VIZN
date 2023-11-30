import cv2 as cv
import numpy as np

def is_rgb(img):
    return len(img.shape) == 3 and img.shape[2] == 3

def detect_skin(image, kernel=np.ones((7,7))):
    """
    Detect skin using HSV values
    Trains an AdaBoost model on the given data.
    Args:
        image (numpy.array): The input to perform skin detection one
        kernel (numpy.array): The kernel used for closing holes in the skin mask 

    Returns:
        skin: an image that only contains components that are skin colored

    """
    # If image is in grayscale return original image as we cannot perform skin detection
    if not is_rgb(image): return image

    # Convert image to HSV format
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Set upper and lower bounds for HSV skin detection
    lower_skin = np.array([0, 25, 10], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin_mask for detection
    skin_mask = cv.inRange(hsv_image, lower_skin, upper_skin)

    # Close holes in skin mask to perserve facial features (ex: teethy smiles and eyes)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)

    # Apply the mask to the image
    skin = cv.bitwise_and(image, image, mask=skin_mask)

    return skin