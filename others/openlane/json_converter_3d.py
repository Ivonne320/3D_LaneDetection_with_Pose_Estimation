# read jsons in a folder and convert them to corresponding format
import os
import json
import numpy as np



pd_jsons_dir = "/home/yiwang/OpenLane/eval/eval_2d_test/training/"
output_dir = "/home/yiwang/OpenLane/eval/eval_data/training/"
depths = '/home/yiwang/ZoeDepth/pred_depth/fine-tuned_for_demo/'
gts = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/'
# pd_jsons = os.listdir(pd_jsons_dir)
segment_jsons = os.listdir(pd_jsons_dir)

for segment in segment_jsons:
    json_files = os.listdir(os.path.join(pd_jsons_dir, segment))
    for pd_json in json_files:
        # get json file
        json_file = os.path.join(pd_jsons_dir, segment, pd_json)
        #  get pd_json basename
        pd_json_basename = os.path.basename(pd_json)
        # retrieve image path
        image_path ="training/" + segment+ "/" + pd_json_basename.split('.')[0] + '.jpg'
        depth = depths + pd_json_basename.split('.')[0] + '.csv'
        np_depth_map = np.loadtxt(depth,delimiter=',')
        gt = gts + segment + '/' + pd_json_basename.split('.')[0] + '.json'
        with open(gt, 'r') as f:
            gt_json = json.load(f)
            camera_intrinsic = gt_json['intrinsic']
            camera_extrinsic = gt_json['extrinsic']
        # print(image_path)
        with open(json_file, 'r') as file:
            input_data = json.load(file)
            img_path = image_path
            output_data = {
                "intrinsic": camera_intrinsic,
                "extrinsic": camera_extrinsic,
                "file_path": img_path,  # Replace <your_image_path> with the actual image path
                "lane_lines": []
            }

            for item in input_data:
                keypoints = item["keypoints"]
                uv = [keypoints[i:i+2] for i in range(0, len(keypoints), 3)]  # Extract u and v
                # extract depth
                u = np.array(keypoints[::3])
                v = np.array(keypoints[1::3])
                
                for k in range(len(u)):
                    if u[k] > 1920 or v[k] > 1080:
                        u[k] = -1
                        v[k] = -1
                    elif u[k] < 0 or v[k] < 0:
                        u[k] = -1
                        v[k] = -1
                u = u[u != -1]
                v = v[v != -1]
                
                z = np.array([np_depth_map[int(v[k]), int(u[k])] for k in range(len(u))])
                # project to 3d
                x = - (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
                y = - (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
                
                xyz = [x.tolist(), z.tolist(),  y.tolist()]
                # transpose xyz
                xyz = list(map(list, zip(*xyz)))
                
                # transpose uv
                uv_transposed = list(map(list, zip(*uv)))
                
                output_data["lane_lines"].append({
                    "xyz": xyz,
                    "category": item.get("category_id", 0),
                    
                })
            output_json = json.dumps(output_data, indent=4)
            output_json_basename = pd_json_basename.split('.')[0] + '.json'
            output_json_path = os.path.join(output_dir, segment, output_json_basename)
            # check if output dir exists
            if not os.path.exists(os.path.dirname(output_json_path)):
                os.makedirs(os.path.dirname(output_json_path))
            with open(output_json_path, 'w') as f:
                f.write(output_json)
    