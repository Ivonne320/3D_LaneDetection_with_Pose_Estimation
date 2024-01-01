import os
import glob
import json
import numpy as np
from PIL import Image

# Parent directories
openlane_json_parent_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/'
openlane_image_parent_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/'

# Output directories
rgb_images_dir = '/home/yiwang/ZoeDepth/openlane/rgb/'
depth_maps_dir = '/home/yiwang/ZoeDepth/openlane/depth/full_trial/'
txt_dir = '/home/yiwang/ZoeDepth/train_test_inputs/'
txt_filepath = os.path.join(txt_dir, 'full_trial.txt')

os.makedirs(depth_maps_dir, exist_ok=True)
os.makedirs(rgb_images_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

segment_dirs = next(os.walk(openlane_json_parent_dir))[1]

with open(txt_filepath, 'a') as txt_file:
    for segment_dir in segment_dirs:
        json_dir_path = os.path.join(openlane_json_parent_dir, segment_dir)
        image_dir_path = os.path.join(openlane_image_parent_dir, segment_dir)

        json_files = sorted(glob.glob(os.path.join(json_dir_path, '*.json')))
        image_files = sorted(glob.glob(os.path.join(image_dir_path, '*jpg')))

        for json_file in json_files[::8]:  # Use your desired step
            with open(json_file, 'r') as f:
                openlane_data = json.load(f)
                # ... your processing code for each json file
                camera_intrinsics = openlane_data["intrinsic"]
                fx = camera_intrinsics[0][0]
                fy = camera_intrinsics[1][1]
                cx = camera_intrinsics[0][2]
                cy = camera_intrinsics[1][2]
            # Create an empty depth map 
            depth_map = np.zeros((1280, 1920))  # Replace with your image dimensions
            
            for lane in openlane_data["lane_lines"]:
            
            # uv_coordinates = lane["uv"]
                depths = lane["xyz"][0]  # Using the x-coordinate as depth
                x_coordinates = lane["xyz"][1]
                y_coordinates = lane["xyz"][2]
                visibility = lane["visibility"]

                x_coordinates = [x_coordinates[i] for i in range(len(x_coordinates)) if visibility[i] == 1.0]
                y_coordinates = [y_coordinates[i] for i in range(len(y_coordinates)) if visibility[i] == 1.0]
                depths = [depths[i] for i in range(len(depths)) if visibility[i] == 1.0]
                points_3D = np.array([x_coordinates, y_coordinates, depths])

                

                # project to image plane
                p = camera_intrinsics @ points_3D
                uv_coordinates = np.zeros((2, len(x_coordinates)))
                uv_coordinates[0] = p[0]/p[2]
                uv_coordinates[1] = p[1]/p[2]
                # discard points where u > 1920 or v > 1280
                uv_coordinates = np.array([uv_coordinates[0], uv_coordinates[1]])
                uv_coordinates = uv_coordinates[:, np.where((uv_coordinates[0] < 1920) & (uv_coordinates[1] < 1280))[0]]
                depths = np.array(depths)[np.where((uv_coordinates[0] < 1920) & (uv_coordinates[1] < 1280))[0]]
            
                
                # Fill in depth values only for visible lane points
                for u, v, depth in zip(uv_coordinates[0], uv_coordinates[1],depths):
                    # if vis == 1.0:  # if the lane point is visible
                    depth_map[int(v)][int(u)] = depth
           

            depth_map = np.flipud(depth_map)
            depth_map = np.fliplr(depth_map)
            print(depth_map.max())
           
            depth_map *= 100
            depth_map = depth_map.astype(np.uint16)
            if depth_map.max() == 0:
                continue
            depth_map[depth_map == 0] = 65535


                # After processing and saving depth_map:
            depth_image_name = os.path.basename(json_file).replace('.json', '.png')
            depth_image_path = os.path.join(depth_maps_dir, depth_image_name)
            # ... save depth map logic
            depth_image = Image.fromarray(depth_map)
            depth_image.save(depth_image_path)                    


            # Find and process corresponding RGB image
            image_prefix = os.path.basename(json_file).replace('.json', '')
            matching_images = [img for img in image_files if image_prefix in img]
            if matching_images:
                rgb_image_path = os.path.join(image_dir_path, os.path.basename(matching_images[0]))
                # ... save RGB image logic
            
            # Write paths and fx to txt file
                txt_file.write(f'{rgb_image_path} {depth_image_path} {fx}\n')
