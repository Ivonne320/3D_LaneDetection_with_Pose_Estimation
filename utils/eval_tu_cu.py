import io
import numpy as np
import torch
import os
import json
# from utils import *
from collections import defaultdict

class JSONLoadError(Exception):
    pass

def compute_f1(gt_bboxs, pred_bboxs, iou_threshold=0.5):
    """
    Computes the Intersection over Union (IoU) between the ground truth (gt) and predicted (pred) bounding boxes.

    Args:
        gt_bboxs (list): A list of ground truth bounding boxes.
        pred_bboxs (list): A list of predicted bounding boxes.
        iou_threshold (float): The IoU threshold for matching gt and pred bounding boxes.

    Returns:
        float: The average IoU between the matched gt and pred bounding boxes.
    """
    total_iou = 0.0
    num_matches = 0 # true positives
    num_gt_lanes = len(gt_bboxs)
    num_pred_lanes = len(pred_bboxs)
    FN = num_gt_lanes # initialize false negatives

    # Iterate over each predicted bounding box and match it with the corresponding ground truth bounding box
    for pred_bbox in pred_bboxs:
        max_iou = 0.0
        matched_gt_bbox = None

        for gt_bbox in gt_bboxs:
            # Compute the IoU between the gt and pred bounding boxes
            iou = compute_bbox_iou(gt_bbox, pred_bbox)

            # If the IoU is greater than the threshold and higher than any previous match, update the match
            if iou > iou_threshold and iou > max_iou:
                max_iou = iou
                matched_gt_bbox = gt_bbox

        # If a match was found, add the IoU to the total and increment the number of matches
        if matched_gt_bbox is not None:
            total_iou += max_iou
            num_matches += 1
            FN -= 1
    
    AP = num_matches / (num_pred_lanes+ 1e-6)
    RC = num_matches / (num_matches + FN + 1e-6)
    f1 = 2 * AP * RC / (AP + RC + 1e-6)

    # Compute the average IoU over all matches
    if num_matches > 0:
        avg_iou = total_iou / num_matches
    else:
        avg_iou = 0.0

    return avg_iou, f1

def compute_lane_accuracy(pred_lane, gt_lane, threshold = 32):
    pred_keyponits = np.array(pred_lane['keypoints']).reshape(-1, 3)
    gt_keyponits = np.array(gt_lane['keypoints']).reshape(-1, 3)
    #get the nearest gt keypoint for each pred keypoint and compute the distance
    gt_nearest_keypoints = []
    for pred_keypoint in pred_keyponits:
        gt_nearest_keypoint = gt_keyponits[np.argmin(np.linalg.norm(gt_keyponits[:, :2] - pred_keypoint[:2], axis=1))]
        gt_nearest_keypoints.append(gt_nearest_keypoint)
    gt_nearest_keypoints = np.array(gt_nearest_keypoints)   
    #compute the distance between pred keypoints and nearest gt keypoints
    dist = np.linalg.norm(pred_keyponits[:, :2] - gt_nearest_keypoints[:, :2], axis=1)
    num_correct = np.sum(dist < threshold)
    accuracy = num_correct / len(pred_keyponits)
    return accuracy

    
def compute_bbox_iou(bbox1, bbox2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1 (list): A list of four numbers representing the coordinates of the first bounding box.
        bbox2 (list): A list of four numbers representing the coordinates of the second bounding box.

    Returns:
        float: The IoU between the two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def CULaneEval(img_pred_lanes, img_gt_lanes):
    pred_bboxs = []
    gt_bboxs = []
    for lane in img_pred_lanes:
        pred_bboxs.append(lane['bbox'])
    for lane in img_gt_lanes:
        gt_bboxs.append(lane['bbox'])
    avg_iou, f1 = compute_f1(gt_bboxs, pred_bboxs)
    if len(img_gt_lanes) == 0 and len(img_pred_lanes) == 0:
        f1 = 1
    return avg_iou, f1

def TuSimpleEval(img_pred_lanes, img_gt_lanes, threshold = 0.85): 
    # intrapolate gt lanes to 100 points
    for lane in img_gt_lanes:
        keypoints = np.array(lane['keypoints']).reshape(-1, 3)
        u = keypoints[:, 0]
        v = keypoints[:, 1]
        visibility = keypoints[:, 2]
        x_new = np.linspace(0, 1, 100)
        u_new = np.interp(x_new, np.linspace(0, 1, len(u)), u)
        v_new = np.interp(x_new, np.linspace(0, 1, len(v)), v)
        visibility_new = np.interp(x_new, np.linspace(0, 1, len(visibility)), visibility)
        # replace keypoints with intrapolated ones
        lane['keypoints'] = np.stack([u_new, v_new, visibility_new], axis=1).reshape(-1).tolist()
        # print('intrapolated lane', lane['keypoints'])
    # TP = 0
    # for pred_lane in img_pred_lanes:
    #     already_matched = False
    #     for gt_lane in img_gt_lanes:
    #         accuracy = compute_lane_accuracy(pred_lane, gt_lane)
    #         if accuracy > threshold and not already_matched:
    #             TP += 1
    #             already_matched = True

    # AP = TP / len(img_pred_lanes)
    # RC = TP / len(img_gt_lanes)
    # f1 = 2 * AP * RC / (AP + RC + 1e-6)
    # return f1
    TP = 0
    FP = 0
    FN = 0
    for pred_lane in img_pred_lanes:
        already_matched = False
        for gt_lane in img_gt_lanes:
            accuracy = compute_lane_accuracy(pred_lane, gt_lane)
            if accuracy > threshold and not already_matched:
                TP += 1
                already_matched = True

        # If no match was found, the predicted lane is a false positive
        if not already_matched:
            FP += 1

    # Iterate over each ground truth lane and count the number of false negatives
    for gt_lane in img_gt_lanes:
        already_matched = False
        for pred_lane in img_pred_lanes:
            accuracy = compute_lane_accuracy(pred_lane, gt_lane)
            if accuracy > threshold and not already_matched:
                already_matched = True

        # If no match was found, the ground truth lane is a false negative
        if not already_matched:
            FN += 1

    AP1 = TP / (TP + FP + 1e-6)
    RC1 = TP / (TP + FN + 1e-6)
    # AP2 = TP / len(img_pred_lanes)
    # RC2 = TP / len(img_gt_lanes)
    f1_1 = 2 * AP1 * RC1 / (AP1 + RC1 + 1e-6)
    # f1_2 = 2 * AP2 * RC2 / (AP2 + RC2 + 1e-6)
    if len(img_gt_lanes) == 0 and len(img_pred_lanes) == 0:
        f1_1 = 1
    return f1_1


        
def main():
    # gt_path = './data_twicedownsampling_sample/annotations/openlane_keypoints_sample_10validation.json'
    # pred_path = './predictions/twice_downsampling-112epoch/jsons/'
    gt_path = './data_twicedownsampling_sample/annotations/openlane_keypoints_sample_10validation.json'
    pred_path = './predictions/twice_downsampling-158epoch/jsons/'
    # Load ground truth
    with open(gt_path) as f:
        gt = json.load(f)
    
    # Load predictions
    pred_files = [f for f in os.listdir(pred_path) if os.path.isfile(os.path.join(pred_path, f))]
    pred_lines = {}
    for pred_file in pred_files:
        pred_json_path = os.path.join(pred_path, pred_file)
        img_id = pred_file.split('.')[0]
        # try:
        #     with open(pred_json_path, 'r') as f:
        #         data_pred = json.load(f)
        # except:
        #     raise JSONLoadError(f"Error loading JSON file: {pred_json_path}")
        with open(pred_json_path, 'r+') as f:
            data = f.read()
            f.seek(0)
            if not data:
                f.write('[]')
                f.seek(0)
            try:
                data_pred = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print('json.decoder.JSONDecodeError: ', e)
                raise JSONLoadError(f"Error loading JSON file: {pred_json_path}")
            if data_pred:
                pred_lines[img_id] = data_pred

    gt_lines = defaultdict(list)
    for img, pred in pred_lines.items():
        for gt_item in gt['annotations']:
            if gt_item['image_id'] == int(img):
                gt_item['pred'] = pred
                print('found pred for img', img)
                gt_lines[str(img)].append(gt_item)
    
    CULane_F1 = []
    TuSimple_F1 = []
    for img, pred in pred_lines.items():
        avg_iou, cu_f1 = CULaneEval(pred, gt_lines[img])
        CULane_F1.append(cu_f1)
        tu_f1 = TuSimpleEval(pred, gt_lines[img])
        TuSimple_F1.append(tu_f1)
    
    print('CULane F1 score: ', np.mean(CULane_F1))
    print('TuSimple F1 score: ', np.mean(TuSimple_F1))



if __name__ == "__main__":
    main()


