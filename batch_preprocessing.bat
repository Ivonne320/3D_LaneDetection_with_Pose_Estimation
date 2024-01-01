#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 64G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mail-user=yihan.wang@epfl.ch 
#SBATCH --mail-type=ALL,TIME_LIMIT


echo STARTING

python3 -m openpifpaf_openlane.openlane_to_coco_48kps \
    --dir_data='/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000' \
    --dir_images='/work/vita/datasets/OpenDriveLab___OpenLane/raw/images' \
    --dir_out='./data_48kps_sample/' --sample \
    
