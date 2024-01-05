### Prepare the prediction json file for evaluation ###

import json
import os
import numpy as np
import glob


pred_2d_path = "/home/yiwang/CIVIL-459-Project/predictions/twice_downsampling-300epoch/val-jsons/demo_2/"
gt_path = "/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/"
segment_names = os.listdir(gt_path)
# print(segment_names)
seg_map = {}
for seg in segment_names:
    seg_map[seg] = []
    #get seg path
    seg_path = os.path.join(gt_path, seg)
    #get all the image names
    img_names = os.listdir(seg_path) 
    for img_name in img_names:
        #get the image name without extension
        img_name = img_name.split('.')[0]
        seg_map[seg].append(img_name)
# print(seg_map)
#iterate through the prediction json folder
json_files = sorted(glob.glob(os.path.join(pred_path, '*.json')))
# print(len(json_files))
for json_file in json_files:
    #get json basename
    json_basename = os.path.basename(json_file).split('.')[0]
    
    #get corresponding segment name
    for seg_name in segment_names:
        image_names = seg_map[seg_name]  
        if json_basename in image_names:
            #make a corresponding segment folder in pred_path and move the json file into it
            seg_pred_path = os.path.join(pred_path, seg_name)
            os.makedirs(seg_pred_path, exist_ok=True)
            #move the json file into the above folder
            os.rename(json_file, os.path.join(seg_pred_path, os.path.basename(json_file)))
            
            break
