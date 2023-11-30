import os
import cv2
import pickle
from config import data_directory, training_directory
from boosting import integral_image, eval_weak_classifier, boosted_predict
from cascade import train_cascade
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


def refine_skin_mask(hsv_image):
    # Define a new, more specific range for skin color
    lower_skin = np.array([0, 58, 30], dtype="uint8")
    upper_skin = np.array([33, 255, 255], dtype="uint8")

    # Create initial skin mask
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # Blur the mask to help remove noise
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)

    return skin_mask



# Detect faces in an image using a trained AdaBoost model
def detect_faces(image, boosted_model, weak_classifiers, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.7, classifier_number=None):
    detected_faces = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to HSV for skin detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    skin_mask = refine_skin_mask(hsv_image)

    # Initial window size should match the trained classifier input
    window_size = (31, 25)
    current_scale = 1.0
    
    # Sliding window approach with scaling
    while window_size[0] * current_scale < gray_image.shape[1] and window_size[1] * current_scale < gray_image.shape[0]:
        scaled_window_size = (int(window_size[0] * current_scale), int(window_size[1] * current_scale))

        for y in range(0, gray_image.shape[0] - scaled_window_size[1], step_size):
            for x in range(0, gray_image.shape[1] - scaled_window_size[0], step_size):
                window = gray_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                resized_window = cv2.resize(window, window_size)
                integral_window = integral_image(resized_window)

                # Classifier evaluation using boosted_predict
                prediction = boosted_predict(np.expand_dims(integral_window, axis=0), boosted_model, weak_classifiers)
                
                if prediction > 0:  # Assuming positive prediction indicates a face
                    face_region = skin_mask[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                    if cv2.countNonZero(face_region) > (0.5 * scaled_window_size[0] * scaled_window_size[1]): 
                        detected_faces.append((x, y, x + scaled_window_size[0], y + scaled_window_size[1]))

        current_scale *= scale_factor

    # Apply Non-Max Suppression to the bounding boxes
    if len(detected_faces) > 0:
        detected_faces = non_max_suppression_fast(np.array(detected_faces), overlapThresh)

    return detected_faces



def load_modelCascade(file_name):
    """
    Loads a trained AdaBoost model from the training directory.
    Args:
        file_name: The name of the file containing the model data.
    Returns:
        The loaded model.
    """
    # Combine the folder path and file name to get the full path
    full_file_path = os.path.join(training_directory, file_name)

    # Use the full path when opening the file
    with open(full_file_path, 'rb') as file:
        model_data = pickle.load(file)
    
    return model_data

#prob should be moved to boosting file but had import problems
def boosted_predict_cascade(images, cascade_dict):
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
    print("Type of cascade:", type(cascade_dict))
    print(f"Debug: Number of images: {len(images)}, Cascade stages: {len(cascade_dict)}")

    if not isinstance(cascade_dict, dict):
        print("Error: Cascade is not a dictionary.")
        return

    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    results = np.zeros(images.shape[0])

    for i in range(images.shape[0]):
        integral_img = integral_image(images[i])
        passed_all_stages = True

        for stage_number, (boosted_model, weak_classifiers) in cascade_dict.items():
            print(f"Processing Stage {stage_number}")
            result = 0
            for classifier_idx, alpha, threshold in boosted_model:
                classifier = weak_classifiers[classifier_idx]
                classifier_response = eval_weak_classifier(classifier, integral_img)

                weak_decision = 1 if classifier_response > threshold else -1
                result += weak_decision * alpha

            if result <= 0:  # Image classified as nonface at this stage
                passed_all_stages = False
                break

        results[i] = 1 if passed_all_stages else -1  # 1 for face, -1 for nonface

    return results



#like detect_faces but for cascades
def detect_faces_cascade(image, cascade, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.7):
    detected_faces = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to HSV for skin detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    skin_mask = refine_skin_mask(hsv_image)

    # Initial window size should match the trained classifier input
    window_size = (31, 25)
    current_scale = 1.0

    # Sliding window approach with scaling
    while window_size[0] * current_scale < gray_image.shape[1] and window_size[1] * current_scale < gray_image.shape[0]:
        scaled_window_size = (int(window_size[0] * current_scale), int(window_size[1] * current_scale))

        for y in range(0, gray_image.shape[0] - scaled_window_size[1], step_size):
            for x in range(0, gray_image.shape[1] - scaled_window_size[0], step_size):
                window = gray_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                resized_window = cv2.resize(window, window_size)
                integral_window = integral_image(resized_window)

                # Classifier evaluation using boosted_predict_cascade
                prediction = boosted_predict_cascade(np.expand_dims(integral_window, axis=0), cascade)

                if prediction > 0:  # Assuming positive prediction indicates a face
                    face_region = skin_mask[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                    if cv2.countNonZero(face_region) > (threshold * scaled_window_size[0] * scaled_window_size[1]): 
                        detected_faces.append((x, y, x + scaled_window_size[0], y + scaled_window_size[1]))

        current_scale *= scale_factor

    # Apply Non-Max Suppression to the bounding boxes
    if len(detected_faces) > 0:
        detected_faces = non_max_suppression_fast(np.array(detected_faces), overlapThresh)
    else:
        print(f"Warning: No prediction returned for window at ({x}, {y})")
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


def main():
   
    

    # Datasets
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    output_dir = os.path.join(data_directory, 'output')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    #Train the model
    model_dataCascade = load_modelCascade("face_detection_cascad.pkl")
    #print(model_dataCascade)
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
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        print(f"Image loaded successfully: {image_path}")

        detected_faces = detect_faces_cascade(image, model_dataCascade)  # Modified to use the cascade
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

            cv2.rectangle(image, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (0, 255, 0), 2)

        fn_face_photos += detected_flags.count(False)

        # Save the image with bounding boxes
        output_path = os.path.join(output_dir, photo_file_name)
        if not cv2.imwrite(output_path, image):
            print(f"Failed to save image: {output_path}")
        else:
            print(f"Image saved successfully: {output_path}")

    # Rest of the code for testing on Cropped Faces and Nonfaces remains unchanged

if __name__ == "__main__":
    main()