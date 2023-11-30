import cv2 as cv
import numpy as np

def is_rgb(img):
    return len(img.shape) == 3 and img.shape[2] == 3

def detect_skin(image, positive_histogram, negative_histogram):
    """
    Detects skin in an image using positive and negative histograms.
    
    Parameters:
        image (numpy.ndarray): The input image array with shape (height, width, 3).
        positive_histogram (numpy.ndarray): The positive histogram.
        negative_histogram (numpy.ndarray): The negative histogram.
        
    Returns:
        numpy.ndarray: The output array with skin detection probabilities, shape (height, width).
    """
    # Make sure image is RGB and not grayscale
    if not is_rgb(image):
        return image
    
    histogram_bins = positive_histogram.shape[0]
    factor = 256 / histogram_bins
    
    # Calculate indices for each color channel
    red_indices = (image[:, :, 0] / factor).astype(int)
    green_indices = (image[:, :, 1] / factor).astype(int)
    blue_indices = (image[:, :, 2] / factor).astype(int)
    
    # Fetch probabilities from histograms using the indices
    skin_values = positive_histogram[red_indices, green_indices, blue_indices]
    non_skin_values = negative_histogram[red_indices, green_indices, blue_indices]
    
    # Compute total probabilities
    total = skin_values + non_skin_values
    
    # Calculate skin probabilities using Bayes rule: P(skin | RGB) = P(RGB | skin) * P(skin) / P(RGB)
    #      skin_vales = P(RGB | skin)
    # non_skin_values = P(RGB | non-skin)
    #  total = P(RGB) = P(RGB | skin) * P(skin) + P(RGB | non-skin) * P(non-skin). For simplicity, we assume P(skin) = P(non-skin) = 0.5
    #          result = P(skin | RGB)
    result = np.divide(skin_values, total, out=np.zeros_like(skin_values), where=total!=0)
    
    return result

import numpy as np


def build_histograms():
    """
    Builds skin and non-skin color histograms from a given dataset file.

    Args:
    - data (numpy.ndarray): The dataset array with shape (N, 4). Each row represents a pixel, and the
                            columns represent the B, G, R values and the label (1 for skin, 2 for non-skin).

    Returns:
    - skin_histogram (numpy.ndarray): A 3D numpy array representing the skin color histogram.
    - nonskin_histogram (numpy.ndarray): A 3D numpy array representing the non-skin color histogram.
    """
    data = np.loadtxt("skin_data/UCI_Skin_NonSkin.txt")
    # Convert data from BGR to RGB order
    rgb_data = data[:, [2, 1, 0, 3]]

    # Seperate skin data from nonskin data
    # Adds only rgb values to appropriate dataset given label
    skin_data = np.array([i[:-1] for i in rgb_data if i[-1] == 1])
    nonskin_data = np.array([i[:-1] for i in rgb_data if i[-1] == 2])

    # Create histograms
    skin_histogram, _ = np.histogramdd(
        skin_data, bins=[32, 32, 32], range=[(0, 256), (0, 256), (0, 256)]
    )
    nonskin_histogram, _ = np.histogramdd(
        nonskin_data, bins=[32, 32, 32], range=[(0, 256), (0, 256), (0, 256)]
    )

    # Normalize histograms
    skin_histogram = skin_histogram / np.sum(skin_histogram)
    nonskin_histogram = nonskin_histogram / np.sum(nonskin_histogram)

    return skin_histogram, nonskin_histogram
