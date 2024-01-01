#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 48G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00

echo STARTING


find /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training/segment-5592790652933523081_667_770_687_770_with_camera_labels/ -path '*.jpg' | xargs -n 1 -P 8 python3 -m openpifpaf.predict --checkpoint /home/yiwang/CIVIL-459-Project/outputs/twice_downsampling_24kps_25/shufflenetv2k16-231127-064149-openlane-slurm1509203.pkl.epoch300 \
--debug-indices cif:0 caf:0 \
--loader-workers=8 --force-complete-pose --instance-threshold=0.05 \
--batch-size=32 \
--long-edge=425 \
--json-output=./predictions/twice_downsampling-300epoch/val-jsons/demo_2 \




