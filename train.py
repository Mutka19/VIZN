import os
import numpy as np
import cv2 as cv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from config import data_directory

def train_model(x_train, y_train):
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), 
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=1.0
        )
    model = clf.fit(x_train, y_train)
    return model

def load_images(dir):
    """ Takes a directory as input """
    """ Returns a list of images"""
    images = []

    for file in os.listdir(dir):
        img = cv.imread(os.path.join(dir, file), cv.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    return images

def resize_images(images, size=(28,28)):
    """ Takes a list of images as input """
    """ Returns a list of resized images """
    images_resized = []
    
    for img in images:
        rsz = cv.resize(img, size).flatten()
        images_resized.append(rsz)

    return images_resized
            

if __name__ == "__main__":
    faces_dir = os.path.join(data_directory, "training_faces")
    nonfaces_dir = os.path.join(data_directory, "training_nonfaces")

    faces = resize_images(load_images(faces_dir))
    nonfaces = resize_images(load_images(nonfaces_dir))

    face_labels = [1] * len(faces)
    non_face_labels = [0] * len(nonfaces)

    x_train = np.array(faces + nonfaces)
    y_train = np.array(face_labels + non_face_labels)

    model = train_model(x_train, y_train)