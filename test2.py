import os
import cv2 as cv
import numpy as np
from config import data_directory
from model import load_model
from testing import detect_faces_cascade, calculate_iou, calculate_precision_recall, test_cropped_faces, test_nonfaces, boosted_predict_cascade
from src.processing import load_test_images
import importlib.util
from src.skin_detection import skin_detect
from src.nms import prepare_boxes, cpu_soft_nms_float, nms_float_fast, nms, normalize_boxes


# Importing face annotations dynamically from a given file path
def import_annotations(annotations_path):
    spec = importlib.util.spec_from_file_location("face_annotations", annotations_path)
    face_annotations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_annotations)
    return face_annotations.annotations


#like detect_faces but for cascades
def detect_faces_cascade(image, cascade, scale_factor=1.25, step_size=5, overlapThresh=0.3, threshold=0.1):
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
                    detected_box = [x, y, x + scaled_window_size[0], y + scaled_window_size[1]]
                    #print("Appending detected box:", detected_box)
                    detected_faces.append(detected_box)
                    detected_scores.append(prediction)

        current_scale *= scale_factor

    #print("detected_faces:", detected_faces)  # Debug output to see the entire list
    # Apply Non-Max Suppression to the bounding boxes
    if len(detected_faces) > 0:
        #detected_faces = non_max_suppression_fast(np.array(detected_faces), overlapThresh)
        # Parameters for NMS

        # print("detected_faces before return:", detected_faces)
        overlapThresh = 0.000000000001
        sigma = 0.2
        min_score = 0.9
        method = "gaussian soft-NMS"

        # Apply NMS
        num_predictions = len(detected_scores)
        # Generate labels as a range from 1 to the number of predictions
        predicted_labels = np.arange(1, num_predictions + 1)
        
        #print(len(detected_scores), type(detected_scores))
        #print(detected_scores[0])

        detected_scores = [item for sublist in detected_scores for item in sublist]
        detected_scores = np.array(detected_scores)
        detected_faces = np.array(detected_faces)
        #print(detected_scores)
        # print(len(detected_scores), type(detected_scores))
        
        #print(detected_scores)
        # print("before NMS", len(detected_faces))
        # print("Len of labels", len(predicted_labels))
        h, w, c = image.shape

      
        normalized_boxesyay = np.array(normalize_boxes(detected_faces, w, h))
        result_boxes, result_scores, result_labels = prepare_boxes(normalized_boxesyay, detected_scores, predicted_labels)
        
        keep_indices  = nms_float_fast(result_boxes, detected_scores, overlapThresh)

        filtered_boxes = result_boxes[keep_indices]
        # Denormalize boxes for drawing
        final_boxes = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            final_boxes.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])

        return final_boxes



def calculate_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculate precision and recall from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall





if __name__ == "__main__":
    # Datasets and model loading
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    output_dir = os.path.join(data_directory, 'output')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    model_dataCascade = load_model()
    model = model_dataCascade['model']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = import_annotations(os.path.join(face_photos_dir, 'face_annotations.py'))
    tp_face_photos, fp_face_photos, fn_face_photos = 0, 0, 0

    # Process each image
    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        true_faces = annotation['faces']
        image_path = os.path.join(face_photos_dir, photo_file_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        detected_faces = detect_faces_cascade(image, model)
        if detected_faces is not None:
            print("after all is said and done we have ", len(detected_faces), " bounding boxes")
            print("first bounding box ", detected_faces[0])
            filtered_detected_faces = [box for box in detected_faces if isinstance(box, (list, np.ndarray)) and len(box) == 4]

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

                cv.rectangle(image, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (0, 255, 0), 2)

            fn_face_photos += detected_flags.count(False)

        output_path = os.path.join(output_dir, photo_file_name)
        cv.imwrite(output_path, image)

    # Calculate precision and recall for face photos
    precision_face_photos, recall_face_photos = calculate_precision_recall(tp_face_photos, fp_face_photos, fn_face_photos)

    # Precision and recall for cropped faces
    tp_cropped, fn_cropped = test_cropped_faces(cropped_faces_dir, model)
    precision_cropped, recall_cropped = calculate_precision_recall(tp_cropped, 0, fn_cropped)

    # Precision and recall for nonfaces
    fp_nonfaces = test_nonfaces(nonfaces_dir, model)
    tn_nonfaces = len(load_test_images(nonfaces_dir)) - fp_nonfaces

    # Output performance metrics
    print("\nDataset: Test Face Photos")
    print(f"True Positives: {tp_face_photos}, False Positives: {fp_face_photos}, False Negatives: {fn_face_photos}")
    print(f"Precision: {precision_face_photos:.2f}, Recall: {recall_face_photos:.2f}")

    print("\nDataset: Test Cropped Faces")
    print(f"True Positives: {tp_cropped}, False Negatives: {fn_cropped}")
    print(f"Precision: {precision_cropped:.2f}, Recall: {recall_cropped:.2f}")

    print("\nDataset: Test Nonfaces")
    print(f"False Positives: {fp_nonfaces}, True Negatives: {tn_nonfaces}")
    print("Precision and recall are not applicable for the nonfaces dataset.")
