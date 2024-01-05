#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 10
#SBATCH --mem 52G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00

echo "STARTING AT NODE $SLURM_NODEID, 48kps, cocosame resumption"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1
srun --gpu-bind=map_gpu:0,1 python3 -u -m openpifpaf.train --ddp --lr=0.00015 --momentum=0.95 --b-scale=10.0 --clip-grad-value=10 --epochs=250 --lr-decay 220 240 --lr-decay-epochs=10 --batch-size=16 --weight-decay=1e-5 --dataset=openlane --openlane-upsample=2 --checkpoint=/home/yiwang/CIVIL-459-Project/outputs/48kps/v16-sample/coco_same/shufflenetv2k16-231122-172541-openlane-slurm1507584.pkl.epoch036 --openlane-orientation-invariant=0.1 --openlane-extended-scale --openlane-train-annotations /home/yiwang/CIVIL-459-Project/data_48kps_sample/annotations/openlane_keypoints_sample_10training.json --openlane-val-annotations /home/yiwang/CIVIL-459-Project/data_48kps_sample/annotations/openlane_keypoints_sample_10validation.json --openlane-train-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training --openlane-val-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation  --output ./outputs/48kps/v16-sample/coco_same/ "$@"