#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 10
#SBATCH --mem 96G
#SBATCH --partition gpu
#SBATCH --gres gpu:4
#SBATCH --time 3-00:00:00

echo "STARTING AT NODE $SLURM_NODEID"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
srun --gpu-bind=closest python3 -u -m openpifpaf.train --ddp --lr=0.001 --momentum=0.9 --b-scale=5.0 --epochs=1000 --lr-warm-up-factor=0.3 --batch-size=32  --val-batches=20 --val-interval=10 --weight-decay=1e-5 --dataset=openlane --openlane-upsample=2 --checkpoint=./outputs/uniform_24kps_25/shufflenetv2k16-231007-222422-openlane-slurm1483249.pkl.epoch011 --openlane-train-annotations ./data_uniform_24kps_quarter/annotations/openlane_keypoints_sample_10training.json --openlane-val-annotations ./data_uniform_24kps_quarter/annotations/openlane_keypoints_sample_10validation.json --openlane-train-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/training --openlane-val-image-dir /work/vita/datasets/OpenDriveLab___OpenLane/raw/images/validation  --loader-workers 1 --output ./outputs/uniform_24kps_25/ "$@"

