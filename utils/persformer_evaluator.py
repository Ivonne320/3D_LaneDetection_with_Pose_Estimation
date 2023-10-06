import eval_2D_lane
from utils import *
import json
import os
import numpy as np
#import openpifpaf

gt_path = '/home/ivonne/3dlane_detection_pifpaf_gr19/data_trial_evaluator/annotations/training/'
sub_gt_dirs = [d for d in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, d))]
pred_json_path = '/home/ivonne/3dlane_detection_pifpaf_gr19/data_trial_evaluator/predictions/training/'
sub_pred_dirs = [d for d in os.listdir(pred_json_path) if os.path.isdir(os.path.join(pred_json_path, d))]   
img_path = '/home/ivonne/3dlane_detection_pifpaf_gr19/data/images/validation/'
sub_img_dirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
evaluator = eval_2D_lane.LaneEval()
pred_lines_sub = []
gt_lines_sub = []

for sub_gt_dir in sub_gt_dirs:
    sub_gt_path = os.path.join(gt_path, sub_gt_dir)
    json_files = [f for f in os.listdir(sub_gt_path) if os.path.isfile(os.path.join(sub_gt_path, f))]
    for json_file in json_files:
        json_path = os.path.join(sub_gt_path, json_file)
        with open(json_path, 'r') as f:
            data_gt = json.load(f)
            gt_lines_sub.append(data_gt)

for sub_pred_dir in sub_pred_dirs:
    sub_pred_path = os.path.join(pred_json_path, sub_pred_dir)
    json_files = [f for f in os.listdir(sub_pred_path) if os.path.isfile(os.path.join(sub_pred_path, f))]
    for json_file in json_files:
        json_path = os.path.join(sub_pred_path, json_file)
        with open(json_path, 'r') as f:
            data_pred = json.load(f)
            if data_pred:
                pred_lines_sub.append(data_pred)
                

output_stats = evaluator.bench_one_submit_openlane_DDP(pred_lines_sub, gt_lines_sub)
print("output_stats: ", output_stats)