import pickle
from config import data_directory, training_directory
import os

def load_model(model_name='face_detection_model.pkl'):
    # Full file path
    full_file_path = os.path.join(training_directory, model_name)

    # Load the model
    with open(full_file_path, 'rb') as file:
        model_data = pickle.load(file)

    return model_data

def load_test_images(directory):
    # Implement a function similar to your training data loading function
    pass


def apply_cascade_to_image(image, cascade):
    # Implement the logic to apply your classifier cascade to the image
    # This should return the detected faces with their bounding boxes
    pass


def evaluate_detections(detections, ground_truth):
    # Implement the logic to calculate false positives and false negatives
    # based on the overlap criterion
    pass


def main():
    pass
    # test_images, ground_truth = load_test_images(data_directory)
    # cascade = load_model()  # Load your trained cascade model

    # for image in test_images:
    #     detections = apply_cascade_to_image(image, cascade)
    #     false_positives, false_negatives = evaluate_detections(detections, ground_truth[image.name])
    #     print(f"Image: {image.name}, False Positives: {false_positives}, False Negatives: {false_negatives}")


if __name__ == "__main__":
    #load the model
    model_data = load_model()
    classifiers = model_data['classifiers']
    extracted_classifiers = model_data['extracted_classifiers']

    print(classifiers, extracted_classifiers)
    main()
