# read jsons in a folder and convert them to corresponding format
import os
import json

pd_jsons_dir = "/home/yiwang/OpenLane/eval/eval_2d_test/training/"
output_dir = "/home/yiwang/OpenLane/eval/eval_data/training/"
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
        image_path ="validation/" + segment+ "/" + pd_json_basename.split('.')[0] + '.jpg'
        # print(image_path)
        with open(json_file, 'r') as file:
            input_data = json.load(file)
            img_path = image_path
            output_data = {
                "file_path": img_path,  # Replace <your_image_path> with the actual image path
                "lane_lines": []
            }

            for item in input_data:
                keypoints = item["keypoints"]
                uv = [keypoints[i:i+2] for i in range(0, len(keypoints), 3)]  # Extract u and v
                # transpose uv
                uv_transposed = list(map(list, zip(*uv)))
                output_data["lane_lines"].append({
                    "category": item.get("category_id", 0),
                    "uv": uv_transposed,
                })
            output_json = json.dumps(output_data, indent=4)
            output_json_basename = pd_json_basename.split('.')[0] + '.json'
            output_json_path = os.path.join(output_dir, segment, output_json_basename)
            # check if output dir exists
            if not os.path.exists(os.path.dirname(output_json_path)):
                os.makedirs(os.path.dirname(output_json_path))
            with open(output_json_path, 'w') as f:
                f.write(output_json)
    