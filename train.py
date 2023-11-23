import os
import numpy as np
import cv2 as cv
from boosting import integral_image
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import adaboost
from config import data_directory

def load_and_process_image(file_path, size=(50, 50)):
    """Load an image, convert it to grayscale, and resize."""
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    img_resized = cv.resize(img, size)
    return img_resized[14:45, 12:37]

def load_images_from_folder(folder_path, size=(50, 50)):
    """Load all images from the specified folder."""
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            images.append(load_and_process_image(file_path, size))
    return np.array(images)

def train_model(faces, nonfaces, weak_count=1000, face_vertical=31, face_horizontal=25, num_classifiers = 15):
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
    for i in range(boosted_classifier):
        boosted_classifier[i][0] = i

    return boosted_classifier, extracted_classifiers
            

if __name__ == "__main__":
    faces_dir = os.path.join(data_directory, "training_faces")
    nonfaces_dir = os.path.join(data_directory, "training_nonfaces")

    faces = load_images_from_folder(faces_dir)
    nonfaces = load_images_from_folder(nonfaces_dir)

    x_train = np.array(faces + nonfaces)
    y_train = np.concatenate((np.ones(len(faces)), np.zeros(len(nonfaces))), axis=0)

    model = train_model(x_train, y_train)