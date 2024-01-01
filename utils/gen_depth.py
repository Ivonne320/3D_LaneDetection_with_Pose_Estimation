import numpy as np
import os
from PIL import Image
import json
import glob

# Directory containing all the OpenLane JSON files
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-16102220208346880_1420_000_1440_000_with_camera_labels/'
openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-54293441958058219_2335_200_2355_200_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-57132587708734824_1020_000_1040_000_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-80599353855279550_2604_480_2624_480_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-141184560845819621_10582_560_10602_560_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-175830748773502782_1580_000_1600_000_with_camera_labels/'
# openlane_json_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000/training/segment-183829460855609442_430_000_450_000_with_camera_labels/'

# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels/'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-16102220208346880_1420_000_1440_000_with_camera_labels/'
openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-54293441958058219_2335_200_2355_200_with_camera_labels'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-57132587708734824_1020_000_1040_000_with_camera_labels/'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-80599353855279550_2604_480_2624_480_with_camera_labels/'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-141184560845819621_10582_560_10602_560_with_camera_labels/'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-175830748773502782_1580_000_1600_000_with_camera_labels/'
# openlane_image_dir = '/work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-183829460855609442_430_000_450_000_with_camera_labels/'

# Directory paths for saving images
rgb_images_dir = '/home/yiwang/ZoeDepth/openlane/rgb/'
# depth_maps_dir = '/home/yiwang/ZoeDepth/openlane/depth/np16/'
depth_maps_dir = './depth_map/'
# txt_dir = '/home/yiwang/ZoeDepth/train_test_inputs/'
txt_dir = './depth_map/'
txt_filepath = os.path.join(txt_dir, 'openlane_training_metric_np16_paths.txt')


os.makedirs(depth_maps_dir, exist_ok=True)
os.makedirs(rgb_images_dir, exist_ok=True)

# Iterate over every 5 JSON file in the directory

json_files = sorted(glob.glob(os.path.join(openlane_json_dir, '*.json')))
image_files = sorted(glob.glob(os.path.join(openlane_image_dir, '*jpg')))
with open(txt_filepath, 'a') as txt_file:
    for json_file in json_files[::5]:
    # for json_file in glob.glob(os.path.join(openlane_json_dir, '*.json')):
        with open(json_file, 'r') as f:
            openlane_data = json.load(f)
            # fus = []
            # fvs = []
            camera_intrinsics = openlane_data["intrinsic"]
            fx = camera_intrinsics[0][0]
            fy = camera_intrinsics[1][1]
            cx = camera_intrinsics[0][2]
            cy = camera_intrinsics[1][2]
            # Create an empty depth map 
        depth_map = np.zeros((1280, 1920))  # Replace with your image dimensions

            # Fill in the depth values from entry["depth_data"] 
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
        
           
            #get x coordinate where visibility is 1
            # x_coordinates = [x_coordinates[i] for i in range(len(x_coordinates)) if visibility[i] == 1.0]
            # y_coordinates = [y_coordinates[i] for i in range(len(y_coordinates)) if visibility[i] == 1.0]
            # depths = [depths[i] for i in range(len(depths)) if visibility[i] == 1.0]
            
            # Fill in depth values only for visible lane points
            for u, v, depth in zip(uv_coordinates[0], uv_coordinates[1],depths):
                # if vis == 1.0:  # if the lane point is visible
                depth_map[int(v)][int(u)] = depth
                # fu = np.abs((u-cx)*depth/x)
                # fv = np.abs((v-cy)*depth/y)
                # fus.append(fu)
                # fvs.append(fv)

        depth_map = np.flipud(depth_map)
        depth_map = np.fliplr(depth_map)
        # Normalize the depth map to the range [0, 255] if necessary
        # depth_map = 255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        
        depth_map =  depth_map.astype(np.uint8)
        # depth_map = depth_map.astype(np.uint16)
        # depth_map[depth_map == 0] = 65535
        
        # Save the depth map as a grayscale image
        depth_image_name = os.path.basename(json_file).replace('.json', '.png')
        depth_image_path = os.path.join(depth_maps_dir, depth_image_name)
        depth_image = Image.fromarray(depth_map)
        depth_image.save(depth_image_path)                    

        # Save the corresponding RGB image
        image_prefix = os.path.basename(json_file).replace('.json', '')
        matching_images = [img for img in image_files if image_prefix in img]
        if matching_images:
            rgb_image_path = os.path.join(openlane_image_dir, os.path.basename(matching_images[0]))
            # with Image.open(matching_images[0]) as rgb_image:
            #     rgb_image.save(rgb_image_path)
        # f_u = np.median(fus)
        # f_v = np.median(fvs)
        # Save the RGB image path, depth map path, focal length to a text file
        txt_file.write(f'{rgb_image_path} {depth_image_path} {fx}\n')

