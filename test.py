import os
import cv2 as cv
from config import data_directory
from src.model import load_model
from src.testing import detect_faces_cascade, calculate_iou, calculate_precision_recall, test_cropped_faces, test_nonfaces
from src.processing import load_test_images
import importlib.util

# Importing face annotations dynamically from a given file path
def import_annotations(annotations_path):
    spec = importlib.util.spec_from_file_location("face_annotations", annotations_path)
    face_annotations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_annotations)
    return face_annotations.annotations


if __name__ == "__main__":
    # Datasets
    face_photos_dir = os.path.join(data_directory, 'test_face_photos')
    output_dir = os.path.join(important_outputs, 'outputBasic')
    cropped_faces_dir = os.path.join(data_directory, 'test_cropped_faces')
    nonfaces_dir = os.path.join(data_directory, 'test_nonfaces')

    #Train the model
    model_dataCascade = load_model()
    model = model_dataCascade['model']
    

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