import numpy as np
import cv2 as cv

def gaussian_probability(mean, std, values):
    """
    Calculate Gaussian probability based on given mean and standard deviation for each value.
    
    Parameters:
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution.
        values (numpy.ndarray): The array of values for which to calculate the Gaussian probabilities.
        
    Returns:
        numpy.ndarray: The array of Gaussian probabilities for the input values.
    """
    return np.exp(-((values - mean)**2) / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))


def get_skin_stats():
    data = np.loadtxt("skin_data/UCI_Skin_NonSkin.txt")
    red = []
    green = []
    blue = []
    for color in data:
        if color[3] == 1:
            red.append(color[2])
            green.append(color[1])
            blue.append(color[0])
    
    red=np.array(red).ravel()
    green=np.array(green).ravel()
    blue=np.array(blue).ravel()

    total = red + green + blue
    red2 = np.divide(red, total, where=total != 0)
    green2 = np.divide(green, total, where=total != 0)

    r_mean = np.mean(red2)
    g_mean = np.mean(green2)
    r_std = np.std(red2)
    g_std = np.std(green2)

    return r_mean, g_mean, r_std, g_std


# Function to fill holes specifically in the head region
def fill_head_holes(mask, head_contour):
    mask_filled = np.zeros_like(mask)

    # Create a filled contour of the head
    cv.drawContours(mask_filled, [head_contour], 0, 255, -1)

    # Apply morphological operations to fill holes in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_filled = cv.morphologyEx(mask_filled, cv.MORPH_CLOSE, kernel)

    return mask_filled


def skin_detect(image):
    # Convert image to float64 to prevent overflow
    image = image.astype(np.float64)
    rows, cols, _ = image.shape

    # Initialize skin detection array
    skin_detection2 = np.zeros((rows, cols))

    # Get mean and std for red and green
    r_mean, g_mean, r_std, g_std = get_skin_stats()

    # Apply probabilistic model using normalized color spaces
    for row in range(rows):
        for col in range(cols):
            red = image[row, col, 0]
            green = image[row, col, 1]
            blue = image[row, col, 2]
        
            total = red + green + blue
            if total > 0:
                r = red / total
                g = green / total
            else:
                r = 0
                g = 0
            
            r_prob = gaussian_probability(r_mean, r_std, r)
            g_prob = gaussian_probability(g_mean, g_std, g)
        
            prob = r_prob * g_prob
            skin_detection2[row, col] = prob

    # Get value of 94.5th percentile of probabilities to use as filter for skin mask
    perc = np.percentile(skin_detection2, 94.45)

    # Apply filter to mask
    mask = (skin_detection2 > perc).astype(np.uint8)

    # Normalize mask
    mask = cv.normalize(mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply threshold to mask image to make it binary
    _, mask_binary = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

    # Apply morphological operations to further refine mask
    kernel = np.ones((3, 3), np.uint8)  
    mask_morph = cv.morphologyEx(mask_binary, cv.MORPH_CLOSE, kernel, iterations=7)
    mask_morph = cv.morphologyEx(mask_morph, cv.MORPH_OPEN, kernel, iterations=3)
    
    # Return image with mask applied to it
    return mask_morph


def get_nonskin_stats():
    data = np.loadtxt("skin_data/UCI_Skin_NonSkin.txt")
    red = []
    green = []
    blue = []
    for color in data:
        if color[3] == 2:
            red.append(color[2])
            green.append(color[1])
            blue.append(color[0])
    
    red=np.array(red).ravel()
    green=np.array(green).ravel()
    blue=np.array(blue).ravel()

    total = red + green + blue
    red2 = np.divide(red, total, where=total != 0)
    green2 = np.divide(green, total, where=total != 0)

    r_mean = np.mean(red2)
    g_mean = np.mean(green2)
    r_std = np.std(red2)
    g_std = np.std(green2)

    return r_mean, g_mean, r_std, g_std


def nonskin_detect(image):
    # Convert image to float64 to prevent overflow
    image = image.astype(np.float64)
    rows, cols, _ = image.shape

    # Initialize skin detection array
    skin_detection2 = np.zeros((rows, cols))

    # Get mean and std for red and green
    r_mean, g_mean, r_std, g_std = get_nonskin_stats()

    # Apply probabilistic model using normalized color spaces
    for row in range(rows):
        for col in range(cols):
            red = image[row, col, 0]
            green = image[row, col, 1]
            blue = image[row, col, 2]
        
            total = red + green + blue
            if total > 0:
                r = red / total
                g = green / total
            else:
                r = 0
                g = 0
            
            r_prob = gaussian_probability(r_mean, r_std, r)
            g_prob = gaussian_probability(g_mean, g_std, g)
        
            prob = r_prob * g_prob
            skin_detection2[row, col] = prob

    # Get value of 80th percentile of probabilities to use as filter for skin mask
    perc = np.percentile(skin_detection2, 80)

    # Convert image back to uint8
    image = image.astype(np.uint8)

    # Apply filter to mask
    mask = (skin_detection2 > (perc)).astype(np.uint8)

    # Normalize mask
    mask = cv.normalize(mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply threshold to mask image to make it binary
    _, mask_binary = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

    # Apply morphological operations to further refine mask
    kernel = np.ones((3,3), np.uint8)
    mask_morph = cv.morphologyEx(mask_binary, cv.MORPH_CLOSE, kernel, iterations=4)
    mask_morph = cv.erode(mask_morph, kernel, iterations=1)
    
    # Return mask
    return mask_morph