import os
from src.cascade import train_cascade
from src.processing import load_faces_from_folder, load_nonfaces_from_folder
from config import data_directory
from src.model import save_model
            

if __name__ == "__main__":
    faces_dir = os.path.join(data_directory, "training_faces")
    nonfaces_dir = os.path.join(data_directory, "training_nonfaces")

    faces = load_faces_from_folder(faces_dir)
    nonfaces = load_nonfaces_from_folder(nonfaces_dir)

    model = train_cascade(faces, nonfaces)

    save_model(model)