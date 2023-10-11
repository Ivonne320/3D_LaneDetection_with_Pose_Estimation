#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 64G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00


echo STARTING

python3 -m openpifpaf_openlane.openlane_to_coco \
    --dir_data='/work/vita/datasets/OpenDriveLab___OpenLane/raw/lane3d_1000' \
    --dir_images='/work/vita/datasets/OpenDriveLab___OpenLane/raw/images' \
    --dir_out='./data_twicedownsampling/' \
    
