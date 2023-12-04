

import numpy as np

def prepare_boxes(boxes, scores, labels):
    result_boxes = boxes.copy()

    cond = (result_boxes < 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates < 0'.format(cond_sum))
        result_boxes[cond] = 0

    cond = (result_boxes > 1)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates > 1. Check that your boxes was normalized at [0, 1]'.format(cond_sum))
        result_boxes[cond] = 1

    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    cond = (area == 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Removed {} boxes with zero area!'.format(cond_sum))
        result_boxes = result_boxes[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]

    return result_boxes, scores, labels



def cpu_soft_nms_float(dets, sc, Nt, sigma, thresh, method):
    """
    Soft-NMS implementation that favors larger boxes.

    :param dets:   boxes format [x1, y1, x2, y2]
    :param sc:     scores for boxes
    :param Nt:     required IoU
    :param sigma:  Gaussian sigma for Soft-NMS
    :param thresh: score threshold for keeping boxes
    :param method: 1 - linear, 2 - gaussian, 3 - original NMS
    :return: indices of boxes to keep
    """

    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # Compute the area of each box
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])

    keep = []
    while dets.shape[0] > 0:
        # Take the box with the highest score
        max_score_index = np.argmax(dets[:, 4])
        max_score_det = dets[max_score_index]
        max_score_area = areas[max_score_index]
        keep.append(max_score_det[-1])

        # Compute IoU of the remaining boxes with the max score box
        xx1 = np.maximum(dets[:, 0], max_score_det[0])
        yy1 = np.maximum(dets[:, 1], max_score_det[1])
        xx2 = np.minimum(dets[:, 2], max_score_det[2])
        yy2 = np.minimum(dets[:, 3], max_score_det[3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas + max_score_area - inter)

        # Suppress boxes that overlap significantly with the max score box
        if method == 1:
            # Linear decay on the scores of overlapping boxes
            weights = np.ones(dets.shape[0])
            weights[ovr > Nt] -= ovr[ovr > Nt]
        elif method == 2:
            # Gaussian decay on the scores of overlapping boxes
            weights = np.exp(-((ovr * ovr) / sigma))
        else:
            # Original NMS
            weights = (ovr <= Nt).astype(float)

        # Make sure 'weights' is a floating-point array
        weights = weights.astype(np.float64)

        # Perform the multiplication
        dets[:, 4] = dets[:, 4] * weights


        #ADDED MAY COMMENT OUT LATER
        # Favor larger boxes by comparing the area
        larger_boxes = np.where((ovr > Nt) & (areas > max_score_area))[0]
        if larger_boxes.size > 0:
            dets[max_score_index, 4] = 0  # Suppress the current max score box
            dets[larger_boxes, 4] = sc[larger_boxes]  # Restore the original scores for larger boxes


        # Keep boxes with a score above the threshold
        remaining_indices = np.where(dets[:, 4] > thresh)[0]
        dets = dets[remaining_indices]
        areas = areas[remaining_indices]

    return keep




def nms_float_fast(dets, scores, thresh):
    """
    # It's different from original nms because we have float coordinates on range [0; 1]
    :param dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
    :param thresh: IoU value for boxes
    :return: index of boxes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_method(boxes, scores, labels, method=3, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    :param boxes: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1] 
    :param scores: list of scores for each model 
    :param labels: list of labels for each model
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :param iou_thr: IoU value for boxes to be a match 
    :param sigma: Sigma value for SoftNMS
    :param thresh: threshold for boxes to keep (important for SoftNMS)
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    """

    # If weights are specified
    if weights is not None:
        if len(boxes) != len(weights):
            print('Incorrect number of weights: {}. Must be: {}. Skip it'.format(len(weights), len(boxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # Do the checks and skip empty predictions
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for i in range(len(boxes)):
        if len(boxes[i]) != len(scores[i]) or len(boxes[i]) != len(labels[i]):
            print('Check length of boxes and scores and labels: {} {} {} at position: {}. Boxes are skipped!'.format(len(boxes[i]), len(scores[i]), len(labels[i]), i))
            continue
        if len(boxes[i]) == 0:
            # print('Empty boxes!')
            continue
        filtered_boxes.append(boxes[i])
        filtered_scores.append(scores[i])
        filtered_labels.append(labels[i])

    # We concatenate everything
    boxes = np.concatenate(filtered_boxes)
    scores = np.concatenate(filtered_scores)
    labels = np.concatenate(filtered_labels)

    # Fix coordinates and removed zero area boxes
    boxes, scores, labels = prepare_boxes(boxes, scores, labels)

    # Run NMS independently for each label
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([l] * len(boxes_by_label))

        if method != 3:
            keep = cpu_soft_nms_float(boxes_by_label.copy(), scores_by_label.copy(), Nt=iou_thr, sigma=sigma, thresh=thresh, method=method)
        else:
            # Use faster function
            keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

        final_boxes.append(boxes_by_label[keep])
        final_scores.append(scores_by_label[keep])
        final_labels.append(labels_by_label[keep])
    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)

    return final_boxes, final_scores, final_labels

def normalize_boxes(boxes, image_width, image_height):
    normalized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_normalized = x1 / image_width
        y1_normalized = y1 / image_height
        x2_normalized = x2 / image_width
        y2_normalized = y2 / image_height
        normalized_boxes.append([x1_normalized, y1_normalized, x2_normalized, y2_normalized])
    return normalized_boxes

def nms(boxes, scores, labels, iou_thr=0.5, weights=None):
    """
    Short call for standard NMS 
    
    :param boxes: 
    :param scores: 
    :param labels: 
    :param iou_thr: 
    :param weights: 
    :return: 
    """
    return nms_method(boxes, scores, labels, method=3, iou_thr=iou_thr, weights=weights)


def soft_nms(boxes, scores, labels, method=2, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    Short call for Soft-NMS
     
    :param boxes: 
    :param scores: 
    :param labels: 
    :param method: 
    :param iou_thr: 
    :param sigma: 
    :param thresh: 
    :param weights: 
    :return: 
    """
    return nms_method(boxes, scores, labels, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh, weights=weights)


