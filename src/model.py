import os
import numpy as np
import cv2 as cv
from src.boosting import integral_image
from src.boosting import generate_classifier
from src.boosting import eval_weak_classifier
from src.boosting import adaboost
import pickle

import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
from config import training_directory

def save_model(model):
    """
    Takes model data and saves it to training directory.
    Args:
        model: Trained adaboost model containing classifier data and their shapes
    """
    model_data = {
        'model': model
    }

    # Specify the file name
    file_name = 'face_detection_model.pkl'

    # Combine the folder path and file name to get the full path
    full_file_path = os.path.join(training_directory, file_name)

    # Use the full path when opening the file
    with open(full_file_path, 'wb') as file:
        pickle.dump(model_data, file)

# Load the trained model from a pickle file
def load_model(model_name='face_detection_model.pkl'):
    # Full file path
    full_file_path = os.path.join(training_directory, model_name)

    # Load the model
    with open(full_file_path, 'rb') as file:
        model_data = pickle.load(file)

    return model_data


def train_model(faces, nonfaces, weak_count=1000, face_vertical=31, face_horizontal=25, num_classifiers = 15):
    """
    Trains an AdaBoost model on the given data.
    Args:
        faces (numpy.array): The input of face images used to train the model
        nonfaces (numpy.array): The input of nonface images used to train the model
        weak_count (int): Number of weak classifiers to be created when training the model
        face_vertical (int): size of each face image in the vertical 'y' direction
        face_horozontal (int): size of each face image in the horozontal 'x' direction
        num_classifiers (int): Number of top classifiers to choose from

    Returns:
        reordered_classifiers: A list of tuples where each tuple contains the index, alpha, and threshold of each classifier
        extracted_classifiers: a list of the top classifiers and their important information in JSON format
    """

    # Generate specified number of weak classifiers
    weak_classifiers = [generate_classifier(face_vertical, face_horizontal) for _ in range(weak_count)]

    # Initialize lists for image integrals
    face_integrals = []
    nonface_integrals = []

    # Compute the integral images for all faces
    for face in faces:
        face_integrals.append(integral_image(face))

    # Compute the integral images for all non-faces
    for nonface in nonfaces:
        nonface_integrals.append(integral_image(nonface))

    # Convert image integral lists into numpy arrays
    face_integrals = np.array(face_integrals)
    nonface_integrals = np.array(nonface_integrals)

    # Concatenate the integral arrays into one along the 1st axis
    examples = np.concatenate((face_integrals, nonface_integrals), axis=0)
    # Create array of labels that matches up with our array of examples
    labels = np.array([1] * len(faces) + [-1] * len(nonfaces))

    # Get number of examples and amount of weak classifiers in order to create a responses array
    example_number = examples.shape[0]
    classifier_number = len(weak_classifiers)

    # Initialize the array to hold the responses, 
    responses = np.zeros((example_number, classifier_number))

    # Loop through each example and classifier in order to avaluate them
    for example in range(example_number):
        integral = examples[example, :, :]
        for feature in range(classifier_number):
            classifier = weak_classifiers[feature]
            responses[example, feature] = eval_weak_classifier(classifier, integral)

    # Run the adaboost algorithms to get the N best classifiers for detecting faces
    boosted_classifier = adaboost(responses, labels, num_classifiers)

    # Get indices for N best best classifiers
    classifiers_indices = [i[0] for i in boosted_classifier]
    extracted_classifiers = [weak_classifiers[i] for i in classifiers_indices]
    reordered_classifiers = []
    for i in range(len(boosted_classifier)):
        reordered_classifiers.append((i, boosted_classifier[i][1], boosted_classifier[i][2]))

    return reordered_classifiers, extracted_classifiers