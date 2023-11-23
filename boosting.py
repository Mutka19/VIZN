import numpy as np
import random


def rectangle_filter1(vertical, horizontal):
    """
    Creates a rectangle filter of type 1
    (white on the left, black on the right).
    Parameters:
    vertical (int): The vertical size of the filter.
    horizontal (int): The horizontal size of the filter.
    Returns:
    numpy.ndarray: The rectangle filter.
    """

    result = np.ones((vertical, 2 * horizontal))
    result[:, horizontal : (2 * horizontal)] = -1
    return result


def rectangle_filter2(vertical, horizontal):
    """
    Creates a rectangle filter of type 2
    (white on the top, black on the bottom).
    Parameters:
    vertical (int): The vertical size of the filter.
    horizontal (int): The horizontal size of the filter.
    Returns:
    numpy.ndarray: The rectangle filter.
    """

    result = np.ones((2 * vertical, horizontal))
    result[vertical : (2 * vertical), :] = -1
    return result


def rectangle_filter3(vertical, horizontal):
    """
    Creates a rectangle filter of type 3
    (white on the left, black in the center, white on the right).
    Parameters:
    vertical (int): The vertical size of the filter.
    horizontal (int): The horizontal size of the filter.
    Returns:
    numpy.ndarray: The rectangle filter.
    """

    result = np.ones((vertical, 3 * horizontal))
    result[:, horizontal : (2 * horizontal)] = -2
    return result


def rectangle_filter4(vertical, horizontal):
    """
    Creates a rectangle filter of type 4
    (white on the top, black in the center, white on the bottom).
    Parameters:
    vertical (int): The vertical size of the filter.
    horizontal (int): The horizontal size of the filter.
    Returns:
    numpy.ndarray: The rectangle filter.
    """

    result = np.ones((3 * vertical, horizontal))
    result[vertical : (2 * vertical), :] = -2
    return result


def rectangle_filter5(vertical, horizontal):
    """
    Creates a rectangle filter of type 5
    (white on the top left, black on the top right,
     black on the bottom left, white on the bottom right).
    Parameters:
    vertical (int): The vertical size of the filter.
    horizontal (int): The horizontal size of the filter.
    Returns:
    numpy.ndarray: The rectangle filter.
    """

    result = np.ones((2 * vertical, 2 * horizontal))
    result[:vertical, horizontal:] = -1
    result[vertical:, :horizontal] = -1
    return result


def integral_image(image):
    # Slow implementation using for loops
    # vertical_size, horizontal_size = image.shape

    # # Initialize the result array with zeros
    # result = np.zeros((vertical_size, horizontal_size), dtype=np.uint64)

    # # Compute sums along horizontal direction
    # for vertical in range(vertical_size):
    #     result[vertical, 0] = image[vertical, 0]
    #     for horizontal in range(1, horizontal_size):
    #         previous_sum = result[vertical, horizontal - 1]
    #         current_value = image[vertical, horizontal]
    #         result[vertical, horizontal] = previous_sum + current_value

    # # Compute sums along vertical direction
    # for horizontal in range(horizontal_size):
    #     for vertical in range(1, vertical_size):
    #         previous_sum = result[vertical - 1, horizontal]
    #         current_value = result[vertical, horizontal]
    #         result[vertical, horizontal] = previous_sum + current_value

    # Fast implementation using numpy.cumsum
    result = np.cumsum(np.cumsum(image, axis=0), axis=1)

    return result


def generate_classifier1(pattern_vertical, pattern_horizontal):
    """
    Generate a random classifier of type 1.
    Parameters:
    pattern_vertical (int): The vertical size of the pattern.
    pattern_horizontal (int): The horizontal size of the pattern.
    Returns:
    The generated classifier.
    """

    size_step = 1.3
    max_vertical = (pattern_vertical - 1) // 1
    max_horizontal = (pattern_horizontal - 1) // 2
    max_vertical_log = int(np.log(max_vertical) / np.log(size_step))
    max_horizontal_log = int(np.log(max_horizontal) / np.log(size_step))

    vertical_size_log = random.randint(4, max_vertical_log)
    horizontal_size_log = random.randint(4, max_horizontal_log)

    vertical_size = int(size_step**vertical_size_log)
    horizontal_size = int(size_step**horizontal_size_log)

    max_vertical_offset = pattern_vertical - (1 * vertical_size) + 1
    max_horizontal_offset = pattern_horizontal - (2 * horizontal_size) + 1

    vertical_offset = random.randint(2, max_vertical_offset)
    horizontal_offset = random.randint(2, max_horizontal_offset)

    positive_rectangles = np.zeros((1, 4), dtype=int)
    positive_rectangles[0, 0] = vertical_offset  # top
    positive_rectangles[0, 1] = vertical_offset + vertical_size - 1  # bottom
    positive_rectangles[0, 2] = horizontal_offset  # left
    positive_rectangles[0, 3] = horizontal_offset + horizontal_size - 1  # right

    negative_rectangles = np.zeros((1, 4), dtype=int)
    negative_rectangles[0, 0] = vertical_offset  # top
    negative_rectangles[0, 1] = vertical_offset + vertical_size - 1  # bottom
    negative_rectangles[0, 2] = horizontal_offset + horizontal_size  # left
    negative_rectangles[0, 3] = horizontal_offset + 2 * horizontal_size - 1  # right

    # Creating the filter
    filter_ = rectangle_filter1(vertical_size, horizontal_size)

    # Creating a dictionary as the result to store all the values
    result = {
        "positive_rectangles": positive_rectangles,
        "negative_rectangles": negative_rectangles,
        "type": 1,
        "negative_value": -1,
        "rectangle_rows": vertical_size,
        "rectangle_cols": horizontal_size,
        "vertical_offset": vertical_offset,
        "horizontal_offset": horizontal_offset,
        "filter": filter_,
    }

    return result


def generate_classifier2(pattern_vertical, pattern_horizontal):
    size_step = 1.3
    max_vertical = (pattern_vertical - 1) // 2
    max_horizontal = (pattern_horizontal - 1) // 1
    max_vertical_log = int(np.log(max_vertical) / np.log(size_step))
    max_horizontal_log = int(np.log(max_horizontal) / np.log(size_step))

    vertical_size_log = random.randint(4, max_vertical_log)
    horizontal_size_log = random.randint(4, max_horizontal_log)

    vertical_size = int(size_step**vertical_size_log)
    horizontal_size = int(size_step**horizontal_size_log)

    max_vertical_offset = pattern_vertical - (2 * vertical_size) + 1
    max_horizontal_offset = pattern_horizontal - (1 * horizontal_size) + 1

    vertical_offset = random.randint(2, max_vertical_offset)
    horizontal_offset = random.randint(2, max_horizontal_offset)

    positive_rectangles = np.zeros((1, 4), dtype=int)
    positive_rectangles[0, 0] = vertical_offset  # top
    positive_rectangles[0, 1] = vertical_offset + vertical_size - 1  # bottom
    positive_rectangles[0, 2] = horizontal_offset  # left
    positive_rectangles[0, 3] = horizontal_offset + horizontal_size - 1  # right

    negative_rectangles = np.zeros((1, 4), dtype=int)
    negative_rectangles[0, 0] = vertical_offset + vertical_size  # top
    negative_rectangles[0, 1] = vertical_offset + 2 * vertical_size - 1  # bottom
    negative_rectangles[0, 2] = horizontal_offset  # left
    negative_rectangles[0, 3] = horizontal_offset + horizontal_size - 1  # right

    # Creating the filter
    filter_ = rectangle_filter2(vertical_size, horizontal_size)

    # Creating a dictionary as the result to store all the values
    result = {
        "positive_rectangles": positive_rectangles,
        "negative_rectangles": negative_rectangles,
        "type": 2,
        "negative_value": -1,
        "rectangle_rows": vertical_size,
        "rectangle_cols": horizontal_size,
        "vertical_offset": vertical_offset,
        "horizontal_offset": horizontal_offset,
        "filter": filter_,
    }

    return result


def generate_classifier3(pattern_vertical, pattern_horizontal):
    size_step = 1.3
    max_vertical = (pattern_vertical - 1) // 1
    max_horizontal = (pattern_horizontal - 1) // 3
    max_vertical_log = int(np.log(max_vertical) / np.log(size_step))
    max_horizontal_log = int(np.log(max_horizontal) / np.log(size_step))

    vertical_size_log = random.randint(4, max_vertical_log)
    horizontal_size_log = random.randint(4, max_horizontal_log)

    vertical_size = int(size_step**vertical_size_log)
    horizontal_size = int(size_step**horizontal_size_log)

    max_vertical_offset = pattern_vertical - vertical_size + 1
    max_horizontal_offset = pattern_horizontal - (3 * horizontal_size) + 1

    vertical_offset = random.randint(2, max_vertical_offset)
    horizontal_offset = random.randint(2, max_horizontal_offset)

    positive_rectangles = np.zeros((2, 4), dtype=int)
    positive_rectangles[0, :] = [
        vertical_offset,
        vertical_offset + vertical_size - 1,
        horizontal_offset,
        horizontal_offset + horizontal_size - 1,
    ]
    positive_rectangles[1, :] = [
        vertical_offset,
        vertical_offset + vertical_size - 1,
        horizontal_offset + 2 * horizontal_size,
        horizontal_offset + 3 * horizontal_size - 1,
    ]

    negative_rectangles = np.zeros((1, 4), dtype=int)
    negative_rectangles[0, :] = [
        vertical_offset,
        vertical_offset + vertical_size - 1,
        horizontal_offset + horizontal_size,
        horizontal_offset + 2 * horizontal_size - 1,
    ]

    # Creating the filter
    filter_ = rectangle_filter3(vertical_size, horizontal_size)

    # Creating a dictionary as the result to store all the values
    result = {
        "positive_rectangles": positive_rectangles,
        "negative_rectangles": negative_rectangles,
        "type": 3,
        "negative_value": -2,
        "rectangle_rows": vertical_size,
        "rectangle_cols": horizontal_size,
        "vertical_offset": vertical_offset,
        "horizontal_offset": horizontal_offset,
        "filter": filter_,
    }

    return result


def generate_classifier4(pattern_vertical, pattern_horizontal):
    size_step = 1.3
    max_vertical = (pattern_vertical - 1) // 3
    max_horizontal = (pattern_horizontal - 1) // 1
    max_vertical_log = int(np.log(max_vertical) / np.log(size_step))
    max_horizontal_log = int(np.log(max_horizontal) / np.log(size_step))

    vertical_size_log = random.randint(4, max_vertical_log)
    horizontal_size_log = random.randint(4, max_horizontal_log)

    vertical_size = int(size_step**vertical_size_log)
    horizontal_size = int(size_step**horizontal_size_log)

    max_vertical_offset = pattern_vertical - (3 * vertical_size) + 1
    max_horizontal_offset = pattern_horizontal - horizontal_size + 1

    vertical_offset = random.randint(2, max_vertical_offset)
    horizontal_offset = random.randint(2, max_horizontal_offset)

    positive_rectangles = np.zeros((2, 4), dtype=int)
    positive_rectangles[0, :] = [
        vertical_offset,
        vertical_offset + vertical_size - 1,
        horizontal_offset,
        horizontal_offset + horizontal_size - 1,
    ]
    positive_rectangles[1, :] = [
        vertical_offset + 2 * vertical_size,
        vertical_offset + 3 * vertical_size - 1,
        horizontal_offset,
        horizontal_offset + horizontal_size - 1,
    ]

    negative_rectangles = np.zeros((1, 4), dtype=int)
    negative_rectangles[0, :] = [
        vertical_offset + vertical_size,
        vertical_offset + 2 * vertical_size - 1,
        horizontal_offset,
        horizontal_offset + horizontal_size - 1,
    ]

    # Creating the filter
    filter_ = rectangle_filter4(vertical_size, horizontal_size)

    # Creating a dictionary as the result to store all the values
    result = {
        "positive_rectangles": positive_rectangles,
        "negative_rectangles": negative_rectangles,
        "type": 4,
        "negative_value": -2,
        "rectangle_rows": vertical_size,
        "rectangle_cols": horizontal_size,
        "vertical_offset": vertical_offset,
        "horizontal_offset": horizontal_offset,
        "filter": filter_,
    }

    return result


def generate_classifier(pattern_vertical, pattern_horizontal):
    """
    Generate a random classifier of type 1, 2, 3, or 4.
    Parameters:
    pattern_vertical (int): The vertical size of the pattern.
    pattern_horizontal (int): The horizontal size of the pattern.
    Returns:
    The generated classifier.
    """
    # Generate a random classifier type between 1 and 4
    classifier_type = random.randint(1, 4)

    if classifier_type == 1:
        result = generate_classifier1(pattern_vertical, pattern_horizontal)
    elif classifier_type == 2:
        result = generate_classifier2(pattern_vertical, pattern_horizontal)
    elif classifier_type == 3:
        result = generate_classifier3(pattern_vertical, pattern_horizontal)
    elif classifier_type == 4:
        result = generate_classifier4(pattern_vertical, pattern_horizontal)

    return result


def rectangle_sum(integral, rectangle, row, col):
    """
    Compute the sum of pixel values within a rectangle using an integral image.
    Parameters:
    integral: 2D numpy array of the integral image of some image A.
    rectangle: list or tuple with elements [top, bottom, left, right].
    row: an offset that is used to adjust the top and bottom.
    col: an offset that is used to adjust the left and right.
    Returns:
    The sum of pixel values within the specified rectangle.
    """
    # Ensure we stay within the boundaries of the image
    top = max(rectangle[0] + row - 2, 0)
    bottom = min(rectangle[1] + row - 2, integral.shape[0] - 1)
    left = max(rectangle[2] + col - 2, 0)
    right = min(rectangle[3] + col - 2, integral.shape[1] - 1)

    if top > bottom or left > right:
        return 0  # Cannot have negative dimensions for the rectangle

    # Calculate the sum within the rectangle
    area1 = integral[top, left]
    area2 = integral[bottom, right]
    area3 = integral[bottom, left]
    area4 = integral[top, right]

    result = area1 + area2 - area3 - area4
    return result


def eval_weak_classifier(classifier, integral, row=1, col=1):
    """
    Computes the response of a weak classifier on an image A,
    on a subwindow whose top left corner is (row, col),
    given the integral image of A.
    """
    # Unpack classifier properties
    positive_rectangles = classifier["positive_rectangles"]
    negative_rectangles = classifier["negative_rectangles"]
    negative_value = classifier["negative_value"]

    # Compute the sum for positive rectangles
    sum_positive = 0
    for rectangle in positive_rectangles:
        sum_positive += rectangle_sum(integral, rectangle, row, col)

    # Compute the sum for negative rectangles
    sum_negative = 0
    for rectangle in negative_rectangles:
        sum_negative += negative_value * rectangle_sum(integral, rectangle, row, col)

    # The total sum is the sum of positive and negative contributions
    total_sum = sum_positive + sum_negative

    return total_sum


def weighted_error(responses, labels, weights, classifier):
    """
    Compute the best threshold for a given classifier,
    along with the corresponding weighted error and alpha value for AdaBoost.
    """
    
    classifier_responses = responses[:, classifier]
    minimum = np.min(classifier_responses)
    maximum = np.max(classifier_responses)
    step = (maximum - minimum) / 50.0
    best_error = 1.0
    best_threshold = None
    best_direction = None
    # Handle case where all responses are the same to avoid errors
    if(maximum == minimum):
        return 0.5, 0, 0
    print(classifier_responses)
    print(f"Minimum: {minimum}, Maximum: {maximum}, Step: {step}")

    for threshold in np.arange(minimum, maximum + step, step):
        thresholded = (classifier_responses > threshold).astype(np.float32)
        thresholded[thresholded == 0] = -1
        error1 = np.sum(weights * (labels != thresholded))
        error = min(error1, 1 - error1)
        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_direction = 1 if error1 < (1 - error1) else -1

    best_alpha = best_direction * 0.5 * np.log((1 - best_error) / best_error)
    if best_error == 0:
        best_alpha = 1

    return best_error, best_threshold, best_alpha


def find_best_classifier(responses, labels, weights):
    """
    Find the best classifier from a set of classifiers based on the weighted error.
    Parameters:
    responses: A matrix where each row contains responses of a training pattern on all weak classifiers.
    labels: Training labels of the patterns.
    weights: Current weights of the patterns according to the AdaBoost algorithm.
    Returns:
    A tuple containing the index of the best classifier, the associated minimum error,
    the best threshold, and the alpha value.
    """
    classifier_number = responses.shape[1]
    best_error = 1.0
    best_threshold = None
    best_classifier = None
    best_alpha = None

    # Iterate through all classifiers
    for classifier in range(classifier_number):
        # Compute the weighted error, threshold, and alpha value for the current classifier
        error, threshold, alpha = weighted_error(responses, labels, weights, classifier)
        if error < best_error:
            best_error = error
            best_threshold = threshold
            best_classifier = classifier
            best_alpha = alpha

    # Return the best classifier index, the associated minimum error, the best threshold, and the alpha value.
    return best_classifier, best_error, best_threshold, best_alpha


def adaboost(responses, labels, rounds):
    """
    Trains an AdaBoost ensemble of decision stumps on the given data.
    Args:
        responses (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
        labels (numpy.ndarray): The target labels of shape (n_samples,).
        rounds (int): The number of boosting rounds to perform.
    Returns:
        list: A list of tuples, where each tuple contains the index of the best feature,
        the weight of the corresponding weak classifier, and the threshold value used to
        split the data.
    """
    example_number = responses.shape[0]

    weights = np.ones(example_number) / example_number
    boosted_responses = np.zeros(example_number)

    result = []

    for round in range(rounds):
        best_classifier, best_error, threshold, alpha = find_best_classifier(
            responses, labels, weights
        )

        result.append((best_classifier, alpha, threshold))

        weak_responses = (responses[:, best_classifier] > threshold).astype(np.float32)
        weak_responses[weak_responses == 0] = -1

        new_weights = weights * np.exp(-alpha * weak_responses * labels)
        new_weights /= np.sum(new_weights)
        weights = new_weights

        boosted_responses += alpha * weak_responses
        thresholded = (boosted_responses > 0).astype(np.float32)
        thresholded[thresholded == 0] = -1
        error = np.mean(thresholded != labels)
        print(f"Round: {round + 1}, Error: {error}, Best Error: {best_error}, "
              f"Best Classifier: {best_classifier}, Alpha: {alpha}, Threshold: {threshold}")


    return result


def boosted_predict(images, boosted_model, weak_classifiers, classifier_number=None):
    """
    Classify a set of instances (images) given the boosted model and the weak classifiers.
    Parameters:
    images - the set of instances for which we want to predict their labels.
             Note: the images size has to be the same as the ones used for training.
    boosted_model - The trained model generated by AdaBoost. Contains tuples with
                    the index, alpha values, and threshold of each weak classifier
                    selected by AdaBoost.
    weak_classifiers - The list of all random weak_classifiers generated.
    classifier_number - How many weak classifiers to use for prediction.
    Return:
    result - The prediction value. Positive values indicate a positive class prediction,
             whereas negative values indicate a negative class prediction.
    """

    if classifier_number is None:
        # Use all weak classifiers selected by AdaBoost
        classifier_number = len(boosted_model)

    # Number of classifiers to use cannot be bigger than how many were selected by AdaBoost
    classifier_number = min(len(boosted_model), classifier_number)

    # If only a single image is provided, convert it to a 3D array of dimension (1, height, width)
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    results = np.zeros(images.shape[0])

    for i in range(images.shape[0]):
        # Calculate the integral image using the provided integral_image function
        integral_img = integral_image(images[i])
        result = 0

        # Iterate through the specified number of classifiers
        for j in range(classifier_number):
            # Access the tuple for each classifier
            classifier_tuple = boosted_model[j]
            classifier_index = classifier_tuple[0]
            classifier_alpha = classifier_tuple[1]
            classifier_threshold = classifier_tuple[2]

            # Retrieve the weak classifier from the list using its index
            weak_cl = weak_classifiers[classifier_index]
            # Evaluate the weak classifier using the integral image
            weak_eval = eval_weak_classifier(weak_cl, integral_img)

            # Apply the threshold to get the weak classifier's decision
            weak_decision = 1 if weak_eval > classifier_threshold else -1
            # Update the result with the weighted decision
            result += weak_decision * classifier_alpha

        results[i] = result

    return results