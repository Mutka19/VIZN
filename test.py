import os
import cv2 as cv
import pickle
from config import data_directory, training_directory
from boosting import boosted_predict
from skin_detection import detect_skin
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
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Initial window size should match the trained classifier input
    window_size = (50, 50)
    current_scale = 1.0
    
def detect_faces(image, boosted_classifiers, extracted_classifiers, scale_factor=1.25, step_size=5):
    # Use skin detection to eliminate potential noise
    rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    skin_img = detect_skin(rgb_img)

    #Convert skin image to grayscale for skin detection
    gray_img = cv.cvtColor(skin_img, cv.COLOR_RGB2GRAY)

    detected_faces = []
    
    # Initial window size should match the trained classifier input
    window_size = (31, 25)

    rows, cols = gray_img.shape

    current_scale = 1.0
    
    # Sliding window approach with scaling
    while window_size[0] * current_scale < cols and window_size[1] * current_scale < rows:
        scaled_window_size = (int(window_size[0] * current_scale), int(window_size[1] * current_scale))
        
        for y in range(0, rows - scaled_window_size[1], step_size):
            for x in range(0, cols - scaled_window_size[0], step_size):
                window = gray_img[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                resized_window = cv.resize(window, window_size)

                score = boosted_predict(resized_window, boosted_classifiers, extracted_classifiers)

                # Face detection based on threshold
                if score > 0:
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
            image = cv.imread(file_path, cv.IMREAD_COLOR)
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
    tp_face_photos, fn_face_photos = 0, 0
    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_path = os.path.join(face_photos_dir, photo_file_name)
        image = cv.imread(image_path)

        if image is not None:
            print("Detecting faces in:", annotation['photo_file_name'])
            detected_faces = detect_faces(image, classifiers, extracted_classifiers)
            if len(detected_faces) >= len(true_faces):
                tp_face_photos += len(true_faces)
            else:
                tp_face_photos += len(detected_faces)
                fn_face_photos += len(true_faces) - len(detected_faces)

    # Testing on Cropped Faces and Nonfaces
    tp_cropped, fn_cropped = test_cropped_faces(cropped_faces_dir, classifiers, extracted_classifiers)
    fp_nonfaces = test_nonfaces(nonfaces_dir, classifiers, extracted_classifiers)
    tn_nonfaces = len(load_test_images(nonfaces_dir)) - fp_nonfaces

    # Output performance metrics
    print("Dataset: Test Face Photos")
    print(f"True Positives: {tp_face_photos}, False Negatives: {fn_face_photos}")
    print("Assumption: This dataset contains only faces.")
    print("\nDataset: Test Cropped Faces")
    print(f"True Positives: {tp_cropped}, False Negatives: {fn_cropped}")
    print("Assumption: This dataset contains only faces.")
    print("\nDataset: Test Nonfaces")
    print(f"False Positives: {fp_nonfaces}, True Negatives: {tn_nonfaces}")
    print("Assumption: This dataset contains only non-faces.")

if __name__ == "__main__":
    main()
