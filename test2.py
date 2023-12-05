import os
import cv2 as cv
import pickle
from config import data_directory, training_directory, important_outputs
from src.boosting import boosted_predict
from train import load_faces_from_folder
import matplotlib.pyplot as plt
import importlib.util
import numpy as np
from src.nms import prepare_boxes, cpu_soft_nms_float, nms_float_fast, nms, normalize_boxes
from src.newSkin import skin_detect
import time
from src.testing import boosted_predict_cascade, calculate_iou, calculate_precision_recall


dataset = 1
image_pathglobal = ""

# Importing face annotations dynamically from a given file path
def import_annotations(annotations_path):
    spec = importlib.util.spec_from_file_location(
        "face_annotations", annotations_path)
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


def test_cropped_faces(directory, cascade):
    """
    Predicts labels for test_cropped_faces dataset
    Parameters:
    - directory: directory containing cropped face images.
    - cascade: a list of cascade stages containing the boosted 
      classifiers and the associated weak classifiers
    Returns:
    - true_positives: number of true positives.
    - false_negatives: number of false negatives.
    """

    # Loads test images from folders
    cropped_images = load_faces_from_folder(directory)

    # Initialize tp and fp values
    true_positives = 0
    false_negatives = 0

    # Loop through each image and predict whether or not it is a face
    for image in cropped_images:
        score = boosted_predict_cascade(image, cascade, 0)
        if score > 0:
            true_positives += 1
        else:
            false_negatives += 1

    return true_positives, false_negatives


def test_nonfaces(directory, cascade):
    """
    Predicts labels for test_nonfaces dataset
    Parameters:
    - directory: directory containing nonface images.
    - cascade: a list of cascade stages containing the boosted 
      classifiers and the associated weak classifiers
    Returns:
    - false_positives: number of false positives.
    """

    # Loads test images from folders
    nonface_images = load_faces_from_folder(directory)

    # Initialize fp value
    false_positives = 0

    # Loop through each face and predict whether or not it is a face
    for image in nonface_images:
        score = boosted_predict_cascade(image, cascade, 0)
        if score > 0:
            false_positives += 1
    return false_positives

# like detect_faces but for cascades
def detect_faces_cascade(image, cascade, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.7):
    detected_faces = []
    detected_scores = []
    # Convert to RGB for skin detection
    gray_image = None
    # Get and apply skin mask
    if dataset == 1 or dataset == 3:
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # print(os.path.basename(image_path), " is the image name")
        fileName = os.path.basename(image_pathglobal)

        # #output_mask_name = f'skin_mask_{image}.JPG'
        output_dir = "important_outputs/skins/" + fileName
        # print(output_dir)
        print(image_pathglobal)
        mask = skin_detect(image_pathglobal, output_dir)
        # mask = cv.imread(output_dir, cv.IMREAD_GRAYSCALE)
        skin_image = cv.bitwise_and(rgb_image, rgb_image, mask=mask)
        gray_image = cv.cvtColor(skin_image, cv.COLOR_RGB2GRAY)

    else:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # _, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(rgb_image)
    # axes[0].axis('off')
    # axes[0].set_title("Original Image")
    # axes[1].imshow(mask, cmap="gray")
    # axes[1].axis('off')
    # axes[1].set_title("Skin Detection Result")
    # plt.tight_layout()
    # plt.show(block=True)

    # Initial window size should match the trained classifier input
    window_size = (25, 31)
    current_scale = 1.0

    # Sliding window approach with scaling
    while window_size[0] * current_scale < gray_image.shape[1] and window_size[1] * current_scale < gray_image.shape[0]:
        scaled_window_size = (
            int(window_size[0] * current_scale), int(window_size[1] * current_scale))

        for y in range(0, gray_image.shape[0] - scaled_window_size[1], step_size):
            for x in range(0, gray_image.shape[1] - scaled_window_size[0], step_size):
                window = gray_image[y:y + scaled_window_size[1],
                                    x:x + scaled_window_size[0]]
                resized_window = cv.resize(window, window_size)

                # Classifier evaluation using boosted_predict_cascade
                prediction = boosted_predict_cascade(resized_window, cascade)

                if prediction > .03:  # Assuming positive prediction indicates a face

                    detected_box = [
                        x, y, x + scaled_window_size[0], y + scaled_window_size[1]]
                    # print("Appending detected box:", detected_box)
                    detected_faces.append(detected_box)
                    detected_scores.append(prediction)

        current_scale *= scale_factor

    # Apply Non-Max Suppression to the bounding boxes
    if detected_faces and len(detected_faces) > 0:

        num_predictions = len(detected_scores)
        predicted_labels = np.arange(1, num_predictions + 1)

        detected_scores = [
            item for sublist in detected_scores for item in sublist]
        detected_scores = np.array(detected_scores)
        detected_faces = np.array(detected_faces)
        print(len(detected_scores), type(detected_scores))

        h, w, c = image.shape

        normalized_boxesyay = np.array(normalize_boxes(detected_faces, w, h))
        result_boxes, result_scores, result_labels = prepare_boxes(
            normalized_boxesyay, detected_scores, predicted_labels)

        keep_indices = cpu_soft_nms_float(
            result_boxes, detected_scores, Nt=0.012, sigma=0.08, thresh=0.15, method=3)
        keep_indices = np.array(keep_indices).astype(
            int)  # Convert to an integer array
        filtered_boxes = result_boxes[keep_indices]
        final_boxes = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            final_boxes.append(
                [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])

        return final_boxes
    else:
        if dataset == 2:
            print("No faces detected in -----test_cropped_faces")
        if dataset == 3:
            print("No faces detected in -----test_nonfaces_photos")

        return []


if __name__ == "__main__":
    # Datasets and model loading
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    output_dir = os.path.join(important_outputs, 'outputAdvanced')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    model_dataCascade = load_model()
    model = model_dataCascade['model']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = import_annotations(os.path.join(
        face_photos_dir, 'face_annotations.py'))
    tp_face_photos, fp_face_photos, fn_face_photos = 0, 0, 0

    # Process each image
    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_pathglobal = os.path.join(face_photos_dir, photo_file_name)
        image = cv.imread(image_pathglobal)

        if image is None:
            print(f"Failed to load image: {image_pathglobal}")
            continue

        dataset = 1
        detected_faces = detect_faces_cascade(image, model)
        # print("after all is said and done we have ", len(detected_faces), " bounding boxes")
        # print("first bounding box ", detected_faces[0])
        filtered_detected_faces = [box for box in detected_faces if isinstance(
            box, (list, np.ndarray)) and len(box) == 4]

        detected_flags = [False] * len(true_faces)

        for detected_box in filtered_detected_faces:
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

            cv.rectangle(image, (detected_box[0], detected_box[1]),
                         (detected_box[2], detected_box[3]), (0, 255, 0), 2)

        fn_face_photos += detected_flags.count(False)

        output_path = os.path.join(output_dir, photo_file_name)
        cv.imwrite(output_path, image)

    # Calculate precision and recall for face photos
    precision_face_photos, recall_face_photos = calculate_precision_recall(
        tp_face_photos, fp_face_photos, fn_face_photos)

    # Precision and recall for cropped faces
    tp_cropped, fn_cropped = test_cropped_faces(cropped_faces_dir, model)
    precision_cropped, recall_cropped = calculate_precision_recall(
        tp_cropped, 0, fn_cropped)

    # # Precision and recall for nonfaces
    fp_nonfaces = test_nonfaces(nonfaces_dir, model)
    tn_nonfaces = len(load_test_images(nonfaces_dir)) - fp_nonfaces

    # Output performance metrics
    print("\nDataset: Test Face Photos")
    print(
        f"True Positives: {tp_face_photos}, False Positives: {fp_face_photos}, False Negatives: {fn_face_photos}")
    print(
        f"Precision: {precision_face_photos:.2f}, Recall: {recall_face_photos:.2f}")

    print("\nDataset: Test Cropped Faces")
    print(f"True Positives: {tp_cropped}, False Negatives: {fn_cropped}")
    print(f"Precision: {precision_cropped:.2f}, Recall: {recall_cropped:.2f}")

    print("\nDataset: Test Nonfaces")
    print(f"False Positives: {fp_nonfaces}, True Negatives: {tn_nonfaces}")
    print("Precision and recall are not applicable for the nonfaces dataset.")