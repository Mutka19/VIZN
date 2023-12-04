import os
import cv2 as cv
import numpy as np

def load_and_process_faces(file_path, size=(50, 50)):
    """Load an image, convert it to grayscale, and resize."""
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    img_resized = cv.resize(img, size)
    return img_resized[14:45, 12:37]


def load_faces_from_folder(folder_path):
    """Load all images from the specified folder."""
    size=(50, 50)
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            images.append(load_and_process_faces(file_path, size))
    return np.array(images)


def load_and_process_nonfaces(file_path, scale=4):
    """Load an image, convert it to grayscale, and extract subwindows."""
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape[:2]
    subwindows = []
    for y in range(0, rows - 31, (31 * scale)):
        for x in range(0, cols - 25, (25 * scale)):
            subwindows.append(img[y:y+31, x:x+25])
    return np.array(subwindows)


def load_nonfaces_from_folder(folder_path):
    """Load all images from the specified folder."""
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            images.extend(load_and_process_nonfaces(file_path))
    return np.array(images)


def load_test_images(directory):
    """
    Loads all images from the specified directory without resizing.
    Args:
        directory (str): Path to the directory containing images.
    Returns:
        List[Tuple[str, numpy.ndarray]]: A list of tuples, where each tuple contains
                                         the file name and the image data as a numpy array.
    """
    images = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(directory, file_name)
            image = cv.imread(file_path, cv.IMREAD_COLOR)
            if image is not None:
                images.append((file_name, image))
    return images