import os
import numpy as np
import cv2 as cv
from boosting import integral_image
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import adaboost
from boosting import boosted_predict
from config import data_directory, training_directory
import pickle

# Datasets
Trainface_photos_dir = os.path.join(data_directory, 'training_faces')
output_dir = os.path.join(data_directory, 'output')
cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
Trainnonfaces_dir = os.path.join(data_directory, 'training_nonfaces')


def save_model(model):
    """
    Takes model data and saves it to the training directory.
    Args:
        model: Trained adaboost model containing classifier data and their shapes
    """
    # Organize the model data into a dictionary
    model_data = {
        'classifiers': model[0],  # Assuming model[0] contains the classifiers
        'extracted_classifiers': model[1],  # Assuming model[1] contains additional classifier information
    }

    # Specify the file name
    file_name = 'face_detection_cascad.pkl'

    # Combine the folder path and file name to get the full path
    full_file_path = os.path.join(training_directory, file_name)

    # Use the full path when opening the file
    with open(full_file_path, 'wb') as file:
        pickle.dump(model_data, file)

def load_images_from_folder(folder, desired_size=(50, 50)):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename), cv.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv.resize(img, desired_size)  # Resize the image
            images.append(img)
    return np.array(images)  # Convert the list of images to a NumPy array


def train_cascade(faces, nonfaces, max_weak_count=1000, stages=5, initial_classifiers=10, classifier_increment=10):
    """
    Trains a cascade of AdaBoost models.
    """
    cascade = []
    face_integrals = np.array([integral_image(face) for face in faces])  # Compute once

    for stage in range(stages):
        print(f"Training stage {stage + 1}/{stages}")
        num_classifiers = initial_classifiers + stage * classifier_increment  # Increase classifiers in later stages
        weak_count = max_weak_count // stages
        weak_classifiers = [generate_classifier(faces.shape[1], faces.shape[0]) for _ in range(weak_count)]

        nonface_integrals = np.array([integral_image(nonface) for nonface in nonfaces])  # Compute for current nonfaces
        if len(nonface_integrals) == 0:
            break
        print("Shape of face_integrals:", face_integrals.shape)
        print("Shape of nonface_integrals:", nonface_integrals.shape)

        examples = np.concatenate((face_integrals, nonface_integrals), axis=0)
        labels = np.array([1] * len(faces) + [-1] * len(nonfaces))

        # Initialize responses array
        responses = np.zeros((examples.shape[0], len(weak_classifiers)))

        # Evaluate classifiers
        for i, integral in enumerate(examples):
            responses[i, :] = [eval_weak_classifier(classifier, integral) for classifier in weak_classifiers]

        # Run AdaBoost for this stage
        boosted_classifier = adaboost(responses, labels, num_classifiers)

        # Extract classifiers for this stage
        cascade.append((boosted_classifier, weak_classifiers))

        if stage < stages - 1:
            # Update nonfaces with false positives for the next stage
            nonfaces = get_false_positives(nonfaces, boosted_classifier, weak_classifiers)

    return cascade

def get_false_positives(nonfaces, boosted_classifier, weak_classifiers):
    false_positives = []
    score = boosted_predict(nonfaces, boosted_classifier, weak_classifiers)
    for i in range(len(score)):
        if(score[i] > 0):
            false_positives.append(nonfaces[i])
        
    return np.array(false_positives)


# Load face and non-face images
faces = load_images_from_folder(Trainface_photos_dir)
nonfaces = load_images_from_folder(Trainnonfaces_dir)

modelCascade = train_cascade(faces, nonfaces)
#print(modelCascade)
save_model(modelCascade)
