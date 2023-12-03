import cv2 as cv
import numpy as np
from src.boosting import boosted_predict
from src.processing import load_faces_from_folder
from src.skin_detection import skin_detect

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
def boosted_predict_cascade(image, cascade, threshold):
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


    for i, stage in enumerate(cascade):
        boosted_classifier, weak_classifiers = stage
        # print(f"Processing Stage {stage_number}")
        score = boosted_predict(image, boosted_classifier, weak_classifiers)
        
        if score <= threshold:
            break

    return score


#like detect_faces but for cascades
def detect_faces_cascade(image, cascade, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.03):
    detected_faces = []
    detected_scores = []
    # Convert to RGB for skin detection
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Get and apply skin mask
    mask = skin_detect(rgb_image)
    skin_image = cv.bitwise_and(rgb_image, rgb_image, mask=mask)
    gray_image = cv.cvtColor(skin_image, cv.COLOR_RGB2GRAY)

    # _, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(rgb_image)
    # axes[0].axis('off')
    # axes[0].set_title("Original Image")
    # axes[1].imshow(skin_image, cmap="gray")
    # axes[1].axis('off')
    # axes[1].set_title("Skin Detection Result")
    # plt.tight_layout()
    # plt.show(block=True)

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
                prediction = boosted_predict_cascade(resized_window, cascade, threshold)

                if prediction > threshold:  # Assuming positive prediction indicates a face
                    # face_region = skin_image[y:y + scaled_window_size[1], x:x + scaled_window_size[0]]
                    # if cv.countNonZero(face_region) > (threshold * scaled_window_size[0] * scaled_window_size[1]): 
                    detected_faces.append((x, y, x + scaled_window_size[0], y + scaled_window_size[1]))
                    detected_scores.append(prediction)

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
