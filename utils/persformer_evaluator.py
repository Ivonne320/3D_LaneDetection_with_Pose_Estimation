import eval_2D_lane
from utils import *
import json
import os
import numpy as np
from collections import deque
#import openpifpaf

# gt_path = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/validation/'
# sub_gt_dirs = [d for d in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, d))]
# pred_json_path = './predictions/uniform-24kps-10epoch/jsons/'
# sub_pred_dirs = [d for d in os.listdir(pred_json_path) if os.path.isdir(os.path.join(pred_json_path, d))]   
# # img_path = '/home/ivonne/3dlane_detection_pifpaf_gr19/data/images/validation/'
# # sub_img_dirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
# evaluator = eval_2D_lane.LaneEval()
# pred_lines_sub = []
# gt_lines_sub = []

# for sub_gt_dir in sub_gt_dirs:
#     sub_gt_path = os.path.join(gt_path, sub_gt_dir)
#     json_files = [f for f in os.listdir(sub_gt_path) if os.path.isfile(os.path.join(sub_gt_path, f))]
#     for json_file in json_files:
#         json_path = os.path.join(sub_gt_path, json_file)
#         with open(json_path, 'r') as f:
#             data_gt = json.load(f)
#             gt_lines_sub.append(data_gt)

# for sub_pred_dir in sub_pred_dirs:
#     sub_pred_path = os.path.join(pred_json_path, sub_pred_dir)
#     json_files = [f for f in os.listdir(sub_pred_path) if os.path.isfile(os.path.join(sub_pred_path, f))]
#     for json_file in json_files:
#         json_path = os.path.join(sub_pred_path, json_file)
#         with open(json_path, 'r') as f:
#             data_pred = json.load(f)
#             if data_pred:
#                 data_pred['file_name'] = json_file.replace('.predictions.json', '')
#                 pred_lines_sub.append(data_pred)
                

# output_stats = evaluator.bench_one_submit_openlane_DDP(pred_lines_sub, gt_lines_sub)
# print("output_stats: ", output_stats)

gt_path = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/validation/'
pred_json_path = './predictions/uniform-24kps-10epoch/jsons/'

evaluator = eval_2D_lane.LaneEval()

def get_json_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.json')]

def load_json_data(json_files):
    data_list = deque()
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            data_list.append(data)
    return list(data_list)

gt_json_files = [f for subdir in os.listdir(gt_path) if os.path.isdir(os.path.join(gt_path, subdir)) for f in get_json_files_in_directory(os.path.join(gt_path, subdir))]
pred_json_files = [f for subdir in os.listdir(pred_json_path) if os.path.isdir(os.path.join(pred_json_path, subdir)) for f in get_json_files_in_directory(os.path.join(pred_json_path, subdir))]

gt_lines_sub = load_json_data(gt_json_files)

pred_lines_sub = deque()
for json_file in pred_json_files:
    with open(json_file, 'r') as f:
        data_pred = json.load(f)
        if data_pred:
            data_pred['file_name'] = os.path.basename(json_file).replace('.predictions.json', '')
            pred_lines_sub.append(data_pred)

output_stats = evaluator.bench_one_submit_openlane_DDP(list(pred_lines_sub), gt_lines_sub)
print("output_stats: ", output_stats)
