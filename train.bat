#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 90G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00


echo STARTING

python3 -m openpifpaf.train --lr=0.0001 --momentum=0.9 --b-scale=5.0 \
--epochs=1000 \
--lr-warm-up-factor=0.01 \
--batch-size=6  --val-interval=1 \
--weight-decay=1e-5 \
--dataset=openlane --openlane-upsample=2 \
--basenet=shufflenetv2k30 --loader-workers=4 \
--openlane-train-annotations /home/yiwang/CIVIL-459-Project/data_twicedownsampling_sample/annotations/openlane_keypoints_sample_10training.json \
--openlane-val-annotations /home/yiwang/CIVIL-459-Project/data_twicedownsampling_sample/annotations/openlane_keypoints_sample_10validation.json \
--openlane-train-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training \
--openlane-val-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation  \
--output ./outputs/no-ddp/


