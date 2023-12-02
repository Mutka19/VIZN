import numpy as np
from boosting import integral_image
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import adaboost
from boosting import boosted_predict

def train_cascade(faces, nonfaces, max_weak_count=1000, stages=10, initial_classifiers=5, classifier_increment=5):
    """
    Trains a cascade of AdaBoost models.
    """
    cascade = []
    face_integrals = np.array([integral_image(face) for face in faces])  # Compute once

    stage = 0

    while stage < stages - 1:
        print(f"Training stage {stage + 1}/{stages}")
        num_classifiers = initial_classifiers + stage * classifier_increment  # Increase classifiers in later stages
        weak_count = max_weak_count // stages
        weak_classifiers = [generate_classifier(faces.shape[1], faces.shape[0]) for _ in range(weak_count)]

        nonface_integrals = np.array([integral_image(nonface) for nonface in nonfaces])  # Compute for current nonfaces

        if len(nonface_integrals) == 0:
            break

        print("Shape of face_integrals:", face_integrals.shape)
        print("Shape of nonface_integrals:", nonface_integrals.shape)

        examples = np.concatenate((face_integrals, nonface_integrals), axis=0)
        labels = np.array([1] * len(faces) + [-1] * len(nonfaces))

        # Initialize responses array
        responses = np.zeros((examples.shape[0], len(weak_classifiers)))

        # Evaluate classifiers
        for i, integral in enumerate(examples):
            responses[i, :] = [eval_weak_classifier(classifier, integral) for classifier in weak_classifiers]

        # Run AdaBoost for this stage
        boosted_classifier = adaboost(responses, labels, num_classifiers)

        if stage < stages - 1:
            # Update nonfaces with false positives for the next stage
            false_positives = get_false_positives(nonfaces, boosted_classifier, weak_classifiers)
            if len(false_positives) < len(nonfaces) / 2.5 or len(false_positives) > len(nonfaces) / 1.05:
                print("Stage training failed to produce classifier! Retrying!")
                continue
            else:
                # Extract classifiers for this stage
                cascade.append((boosted_classifier, weak_classifiers))
                nonfaces = false_positives
                stage+=1

    return cascade


def get_false_positives(nonfaces, boosted_classifier, weak_classifiers):
    false_positives = []
    score = boosted_predict(nonfaces, boosted_classifier, weak_classifiers)
    for i in range(len(score)):
        if(score[i] > 0):
            false_positives.append(nonfaces[i])
        
    return np.array(false_positives)