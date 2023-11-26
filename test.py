import os
import cv2
import pickle
from config import data_directory
from boosting import integral_image, eval_weak_classifier, adaboost
import importlib.util

# Importing face annotations dynamically from a given file path
def import_annotations(annotations_path):
    spec = importlib.util.spec_from_file_location("face_annotations", annotations_path)
    face_annotations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_annotations)
    return face_annotations.annotations

# Load the trained model from a pickle file
def load_model(model_name='face_detection_model.pkl'):
    # Path to the Training folder
    folder_path = 'Training'  # or './Training'

    # Full file path
    full_file_path = os.path.join(folder_path, model_name)

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

def calculate_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculate precision and recall from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall


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
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is not None:
                images.append((file_name, image))
    return images


def test_cropped_faces(directory, classifiers, extracted_classifiers):
    cropped_images = load_test_images(directory)
    true_positives = 0
    false_negatives = 0

    for file_name, image in cropped_images:
        detected_faces = detect_faces(image, classifiers, extracted_classifiers)
        if len(detected_faces) > 0:
            true_positives += 1
        else:
            false_negatives += 1

    return true_positives, false_negatives

def test_nonfaces(directory, classifiers, extracted_classifiers):
    nonface_images = load_test_images(directory)
    false_positives = 0

    for file_name, image in nonface_images:
        detected_faces = detect_faces(image, classifiers, extracted_classifiers)
        if len(detected_faces) > 0:
            false_positives += 1

    return false_positives


# The main function processes each annotated image and evaluates the detections
def main():
    # Load model data
    model_data = load_model()
    classifiers = model_data['classifiers']
    extracted_classifiers = model_data['extracted_classifiers']

    # Datasets
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    # Testing on Face Photos
    annotations = import_annotations(os.path.join(face_photos_dir, 'face_annotations.py'))
    tp_face_photos, fp_face_photos, fn_face_photos = 0, 0, 0
    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_path = os.path.join(face_photos_dir, photo_file_name)
        image = cv2.imread(image_path)

        if image is not None:
            detected_faces = detect_faces(image, classifiers, extracted_classifiers)
            detected_flags = [False] * len(true_faces)

            for detected_box in detected_faces:
                match_found = False
                for idx, true_box in enumerate(true_faces):
                    iou = calculate_iou(detected_box, true_box)
                    if iou > 0.5:
                        tp_face_photos += 1
                        detected_flags[idx] = True
                        match_found = True
                        break
                if not match_found:
                    fp_face_photos += 1

            fn_face_photos += detected_flags.count(False)

    # Testing on Cropped Faces and Nonfaces
    tp_cropped, fn_cropped = test_cropped_faces(cropped_faces_dir, classifiers, extracted_classifiers)
    fp_nonfaces = test_nonfaces(nonfaces_dir, classifiers, extracted_classifiers)
    tn_nonfaces = len(load_test_images(nonfaces_dir)) - fp_nonfaces  # Calculate True Negatives in test_nonfaces

    # Output performance metrics
    print("Dataset: Test Face Photos")
    print(f"True Positives: {tp_face_photos}, False Positives: {fp_face_photos}, False Negatives: {fn_face_photos}")
    print("Assumption: This dataset contains both faces and non-faces.")
    print("\nDataset: Test Cropped Faces")
    print(f"True Positives: {tp_cropped}, False Negatives: {fn_cropped}, False Positives: 0, True Negatives: 0")
    print("Assumption: This dataset contains only faces. No false positives or true negatives expected.")
    print("\nDataset: Test Nonfaces")
    print(f"False Positives: {fp_nonfaces}, True Negatives: {tn_nonfaces}, True Positives: 0, False Negatives: 0")
    print("Assumption: This dataset contains only non-faces. No true positives or false negatives expected.")

if __name__ == "__main__":
    main()

