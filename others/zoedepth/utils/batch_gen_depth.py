import os
import glob
import json
import torch
import numpy as np
from PIL import Image

# repo = "isl-org/ZoeDepth"
# model_zoe_nk_origin=torch.hub.load(repo, "ZoeD_NK", pretrained=True)
# model_zoe_nk_origin=model_zoe_nk_origin.to("cuda")

# Parent directories
openlane_json_parent_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/'
openlane_image_parent_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/'

# Output directories
rgb_images_dir = '/home/yiwang/ZoeDepth/openlane/rgb/'
depth_maps_dir = '/home/yiwang/ZoeDepth/openlane/depth/overfitting/'
txt_dir = '/home/yiwang/ZoeDepth/train_test_inputs/'
txt_filepath = os.path.join(txt_dir, 'overfitting.txt')

os.makedirs(depth_maps_dir, exist_ok=True)
os.makedirs(rgb_images_dir, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

segment_dirs = next(os.walk(openlane_json_parent_dir))[1]
counter = 0

with open(txt_filepath, 'a') as txt_file:
    for segment_dir in segment_dirs:
        json_dir_path = os.path.join(openlane_json_parent_dir, segment_dir)
        image_dir_path = os.path.join(openlane_image_parent_dir, segment_dir)

        json_files = sorted(glob.glob(os.path.join(json_dir_path, '*.json')))
        image_files = sorted(glob.glob(os.path.join(image_dir_path, '*jpg')))
        

        for json_file in json_files[::10]:  # Use your desired step
            
            with open(json_file, 'r') as f:
                openlane_data = json.load(f)
                # ... your processing code for each json file
                camera_intrinsics = openlane_data["intrinsic"]
                fx = camera_intrinsics[0][0]
                fy = camera_intrinsics[1][1]
                cx = camera_intrinsics[0][2]
                cy = camera_intrinsics[1][2]
            # Create a depth map pre-filled with original model predictions
            # find corresponding RGB image in image_dir_path
            image_prefix = os.path.basename(json_file).replace('.json', '')
            matching_images = [img for img in image_files if image_prefix in img]
            if matching_images:
                rgb_image_path = os.path.join(image_dir_path, os.path.basename(matching_images[0]))
                
            rgb_img = Image.open(rgb_image_path).convert('RGB')
            # depth_map = model_zoe_nk_origin.infer_pil(rgb_img)
            depth_map = np.zeros((1280, 1920))
            
            # change pixels with gt to depth_gt
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
                    # depth_map[(1280-int(v))][(1920-int(u))] = depth

            depth_map = np.flipud(depth_map)
            depth_map = np.fliplr(depth_map)
           
            # mask = np.where(depth_zero != 0)
            # depth_map = depth_map + depth_zero
           
            depth_map *= 100
            depth_map = depth_map.astype(np.uint16)
            if depth_map.max() <= 15000:
                continue
            print(depth_map.max()/100) 
            depth_map[depth_map == 0] = 65535
           

                # After processing and saving depth_map:
            depth_image_name = os.path.basename(json_file).replace('.json', '.png')
            depth_image_path = os.path.join(depth_maps_dir, depth_image_name)
            # ... save depth map logic
            depth_image = Image.fromarray(depth_map)
            depth_image.save(depth_image_path)   
            counter += 1                 


            # Find and process corresponding RGB image
            image_prefix = os.path.basename(json_file).replace('.json', '')
            matching_images = [img for img in image_files if image_prefix in img]
            if matching_images:
                rgb_image_path = os.path.join(image_dir_path, os.path.basename(matching_images[0]))
                # ... save RGB image logic
            
            # Write paths, fx and mask to txt file in one line
                txt_file.write(f"{rgb_image_path} {depth_image_path} {fx}\n")
                print("txt file written")
                
            if counter > 600:
                print(f"Terminating after processing {counter} items.")
                break
        if counter > 600:
            break

        

        
print("Script completed.")    