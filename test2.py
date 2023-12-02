import os
import cv2 as cv
import pickle
from config import data_directory, training_directory
from boosting import boosted_predict
from skin_detection import skin_detect
from train import load_faces_from_folder
import matplotlib.pyplot as plt
import importlib.util
import numpy as np

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
    cropped_images = load_faces_from_folder(directory)
    true_positives = 0
    false_negatives = 0

    for image in cropped_images:
        detected_faces = detect_faces_cascade(image, cascade)
        if len(detected_faces) > 0:
            true_positives += 1
        else:
            false_negatives += 1

    return true_positives, false_negatives

def test_nonfaces(directory, cascade):
    nonface_images = load_faces_from_folder(directory)
    false_positives = 0

    for image in nonface_images:
        detected_faces = detect_faces_cascade(image, cascade)
        if len(detected_faces) > 0:
            false_positives += 1

    return false_positives


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


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    print(f"Before NMS: {len(boxes)} boxes")
    # initialize the list of picked indexes    
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))


    # return only the bounding boxes that were picked using the integer data type
    print(f"After NMS: {len(pick)} boxes")
    return boxes[pick].astype("int")


#prob should be moved to boosting file but had import problems
def boosted_predict_cascade(image, cascade):
    """
    Classify a set of instances (images) using a cascade of boosted models.
    Parameters:
    - images: numpy.ndarray, an array of instances for classification.
    - cascade_dict: dict, a dictionary where each key-value pair is a stage in the cascade, 
      with the key being the stage number and the value being a tuple of a boosted model 
      and its corresponding weak classifiers.
    Returns:
    - results: numpy.ndarray, the prediction results for each image.
    """

    passed_all_stages = True

    for i, stage in enumerate(cascade):
        boosted_classifier, weak_classifiers = stage
        # print(f"Processing Stage {stage_number}")
        score = boosted_predict(image, boosted_classifier, weak_classifiers)
        
        if score < 0:
            passed_all_stages = False
            break

    return passed_all_stages


#like detect_faces but for cascades
def detect_faces_cascade(image, cascade, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.7):
    detected_faces = []
    # Convert to RGB for skin detection
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Get and apply skin mask
    skin_image = skin_detect(rgb_image)
    gray_image = cv.cvtColor(skin_image, cv.COLOR_RGB2GRAY)

    # _, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(rgb_image)
    # axes[0].axis('off')
    # axes[0].set_title("Original Image")
    # axes[1].imshow(skin_image, cmap="gray")
    # axes[1].axis('off')
    # axes[1].set_title("Skin Detection Result")
    # plt.tight_layout()
    # plt.show(block=False)

    # Initial window size should match the trained classifier input
    window_size = (25, 31)
    current_scale = 1.0

    # Sliding window approach with scaling
    while window_size[0] * current_scale < gray_image.shape[1] and window_size[1] * current_scale < gray_image.shape[0]:
        scaled_window_size = (int(window_size[0] * current_scale), int(window_size[1] * current_scale))

        for y in range(0, gray_image.shape[0] - scaled_window_size[1], step_size):
            for x in range(0, gray_image.shape[1] - scaled_window_size[0], step_size):
                window = gray_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                resized_window = cv.resize(window, window_size)

                # Classifier evaluation using boosted_predict_cascade
                prediction = boosted_predict_cascade(resized_window, cascade)

                if prediction:  # Assuming positive prediction indicates a face
                    # face_region = skin_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                    # if cv.countNonZero(face_region) > (threshold * scaled_window_size[0] * scaled_window_size[1]): 
                    detected_faces.append((x, y, x + scaled_window_size[0], y + scaled_window_size[1]))

        current_scale *= scale_factor

    # Apply Non-Max Suppression to the bounding boxes
    if len(detected_faces) > 0:
        detected_faces = non_max_suppression_fast(np.array(detected_faces), overlapThresh)
    # else:
    #     print(f"Warning: No prediction returned for window at ()")
    return detected_faces


def calculate_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculate precision and recall from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall


if __name__ == "__main__":
    # Datasets
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    output_dir = os.path.join(data_directory, 'output')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    #Train the model
    model_dataCascade = load_model()
    model = model_dataCascade['model']
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Import annotations
    annotations = import_annotations(os.path.join(face_photos_dir, 'face_annotations.py'))
    tp_face_photos, fp_face_photos, fn_face_photos = 0, 0, 0

    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_path = os.path.join(face_photos_dir, photo_file_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        print(f"Image loaded successfully: {image_path}")

        detected_faces = detect_faces_cascade(image, model)  # Modified to use the cascade
        detected_flags = [False] * len(true_faces)

        # Draw bounding boxes and count TP, FP, FN
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

            cv.rectangle(image, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (0, 255, 0), 2)

        fn_face_photos += detected_flags.count(False)

        # Save the image with bounding boxes
        output_path = os.path.join(output_dir, photo_file_name)
        if not cv.imwrite(output_path, image):
            print(f"Failed to save image: {output_path}")
        else:
            print(f"Image saved successfully: {output_path}")

    tp_cropped, fn_cropped = test_cropped_faces(cropped_faces_dir, model)
    fp_nonfaces = test_nonfaces(nonfaces_dir, model)
    tn_nonfaces = len(load_test_images(nonfaces_dir)) - fp_nonfaces  # Calculate True Negatives in test_nonfaces

    # Calculate precision and recall for face photos
    precision_face_photos, recall_face_photos = calculate_precision_recall(tp_face_photos, fp_face_photos, fn_face_photos)

    # Calculate precision and recall for cropped faces
    precision_cropped, recall_cropped = calculate_precision_recall(tp_cropped, 0, fn_cropped)  # FP is 0 for cropped faces

    # Output performance metrics
    print("Dataset: Test Face Photos")
    print(f"True Positives: {tp_face_photos}, False Positives: {fp_face_photos}, False Negatives: {fn_face_photos}")
    print(f"Precision: {precision_face_photos:.2f}, Recall: {recall_face_photos:.2f}")
    print("Assumption: This dataset contains both faces and non-faces.")
    
    print("\nDataset: Test Cropped Faces")
    print(f"True Positives: {tp_cropped}, False Negatives: {fn_cropped}, False Positives: 0")
    print(f"Precision: {precision_cropped:.2f}, Recall: {recall_cropped:.2f}")
    print("Assumption: This dataset contains only faces. No false positives or true negatives expected.")
    
    print("\nDataset: Test Nonfaces")
    print(f"False Positives: {fp_nonfaces}, True Negatives: {tn_nonfaces}")
    print("Assumption: This dataset contains only non-faces. No true positives or false negatives expected.")