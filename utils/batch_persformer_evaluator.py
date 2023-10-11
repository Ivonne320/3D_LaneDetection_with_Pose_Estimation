import eval_2D_lane
from utils import *
import json
import os
import re
import numpy as np
from collections import deque

class JSONLoadError(Exception):
    pass

def extract_order_number(filename):
    # Extract the number from the filename. 
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

# Create a dictionary to store GT paths with order numbers as keys
gt_mapping = {}
gt_parent_dir = "/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/validation/"

# Traverse GT parent directories
for subdir in os.listdir(gt_parent_dir):
    subdir_path = os.path.join(gt_parent_dir, subdir)
    if os.path.isdir(subdir_path) and subdir.startswith("segment"):
        for fname in os.listdir(subdir_path):
            order_number = extract_order_number(fname)
            if order_number != -1:
                gt_mapping[order_number] = os.path.join(subdir_path, fname)


## Pair Predictions with GT
# Create lists to store paths
gt_list = []
pred_list = []

# Get all prediction paths
prediction_parent_dir = "./predictions/2kps-117epoch/jsons/"
all_pred_paths = [os.path.join(prediction_parent_dir, fname) for fname in os.listdir(prediction_parent_dir) if fname.endswith('.predictions.json')]
# Sort them based on order number
all_pred_paths.sort(key=extract_order_number)

for pred_path in all_pred_paths:
    order_number = extract_order_number(os.path.basename(pred_path))
    gt_path = gt_mapping.get(order_number)
    if gt_path:  # Check if the GT file exists in the mapping
        gt_list.append(gt_path)
        pred_list.append(pred_path)

# Batch processing
def process_batch(gt_batch, pred_batch, evaluator):
    
    gt_sub_lines = []
    pred_sub_lines = []
    for gt_path, pred_path in zip(gt_batch, pred_batch):
        with open(gt_path, 'r') as gt_file, open(pred_path, 'r+') as pred_file:
            gt_data = json.load(gt_file)
            gt_sub_lines.append(gt_data)
            data_pred_reprocess = pred_file.read()
            pred_file.seek(0)
            if not data_pred_reprocess:
                pred_file.write('[]')
                pred_file.seek(0)
            try:
                pred_data = json.load(pred_file)
            except json.decoder.JSONDecodeError as e:
                print('json.decoder.JSONDecodeError: ', e)
                raise JSONLoadError(f"Error loading JSON file: {pred_path}")
          
            if pred_data:
                # Split the path into its components
                parts = os.path.normpath(gt_path).split(os.sep)

                # Find the index of "validation" in the path components
                validation_index = parts.index('validation') if 'validation' in parts else -1

                # Get the part of the path from "validation" onwards
                new_path = os.sep.join(parts[validation_index:])
                # Replace the ".json" extension with ".jpg"
                new_path = new_path.replace('.json', '.jpg')
                pred_data[0]['file_name'] = new_path
                pred_sub_lines.append(pred_data)
            
            result = evaluator.bench_one_submit_openlane_DDP(pred_sub_lines, gt_sub_lines)
            
    return result

BATCH_SIZE = 60  # You can set this to the desired batch size

results = []
for i in range(0, len(gt_list), BATCH_SIZE):
    gt_batch = gt_list[i:i+BATCH_SIZE]
    pred_batch = pred_list[i:i+BATCH_SIZE]
    evaluator = eval_2D_lane.LaneEval()
    batch_results = process_batch(gt_batch, pred_batch, evaluator)
    results.append(np.array(batch_results))
    print("Batch Results: ", batch_results)

print("Results: ", results)
