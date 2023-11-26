import os
import cv2
import pickle
from config import data_directory, training_directory
from boosting import integral_image, eval_weak_classifier
import importlib.util

# Importing face annotations dynamically from a given file path
def import_annotations(annotations_path):
    spec = importlib.util.spec_from_file_location("face_annotations", annotations_path)
    face_annotations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_annotations)
    return face_annotations.annotations

# Load the trained model from a pickle file
def load_model(model_name='face_detection_model.pkl'):
    # Full file path
    full_file_path = os.path.join(training_directory, model_name)

    # Load the model
    with open(full_file_path, 'rb') as file:
        model_data = pickle.load(file)

    return model_data

# Calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(boxA, boxB):
    # Coordinate calculations for intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Areas of individual boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

# Detect faces in an image using a trained AdaBoost model
def detect_faces(image, classifiers, extracted_classifiers, scale_factor=1.25, step_size=5):
    detected_faces = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initial window size should match the trained classifier input
    window_size = (50, 50)
    current_scale = 1.0
    
    # Sliding window approach with scaling
    while window_size[0] * current_scale < gray_image.shape[1] and window_size[1] * current_scale < gray_image.shape[0]:
        scaled_window_size = (int(window_size[0] * current_scale), int(window_size[1] * current_scale))
        
        for y in range(0, gray_image.shape[0] - scaled_window_size[1], step_size):
            for x in range(0, gray_image.shape[1] - scaled_window_size[0], step_size):
                window = gray_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                resized_window = cv2.resize(window, window_size)
                integral_window = integral_image(resized_window)
                
                # Classifier evaluation
                sum_alpha = 0
                for classifier, alpha, threshold in classifiers:
                    feature = eval_weak_classifier(extracted_classifiers[classifier], integral_window)
                    sum_alpha += alpha * feature
                
                # Face detection based on threshold
                if sum_alpha >= threshold:
                    detected_faces.append((x, y, x + scaled_window_size[0], y + scaled_window_size[1]))
        
        current_scale *= scale_factor
    
    # TODO: Implement Non-Max Suppression to handle overlapping detections
    return detected_faces

# The main function processes each annotated image and evaluates the detections
def main():
    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0

    # Load annotations and model data
    annotations = import_annotations(os.path.join(data_directory, 'test_face_photos', 'face_annotations.py'))
    model_data = load_model()
    classifiers = model_data['classifiers']
    extracted_classifiers = model_data['extracted_classifiers']

    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_path = os.path.join(data_directory, 'test_face_photos', photo_file_name)
        image = cv2.imread(image_path)

        if image is not None:
            detected_faces = detect_faces(image, classifiers, extracted_classifiers)
            detected_flags = [False] * len(true_faces)

            for detected_box in detected_faces:
                match_found = False
                for idx, true_box in enumerate(true_faces):
                    iou = calculate_iou(detected_box, true_box)
                    if iou > 0.5:
                        true_positive_count += 1
                        detected_flags[idx] = True
                        match_found = True
                        break
                
                if not match_found:
                    false_positive_count += 1

            false_negative_count += detected_flags.count(False)
        else:
            print(f"Failed to load image: {photo_file_name}")

    # Output performance metrics
    print(f"True Positives: {true_positive_count}")
    print(f"False Positives: {false_positive_count}")
    print(f"False Negatives: {false_negative_count}")

if __name__ == "__main__":
    main()
