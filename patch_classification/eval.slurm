#!/bin/bash
#SBATCH --job-name=NN-FINN
#SBATCH -o  %J-22-1.out -e  %J-22-1.err
#NC="-c 1"
#NP="-n 1"
#REQUEST="--mem=4000"
#SBATCH --gres=gpu:K20Xm:1
#SBATCH -t 300

module load gcc/latest
module load nvidia/7.5
module load cudnn/7.5-v5

python train.py --function train_net --data_path tif/ --save_dir hyper_resnet_models --batch_size 64 --lr 0.00001 --log_after 20 --cuda 1 --device 0 --pretrained_model hyper_resnet_models/model-93.pt
##python run_model.py resnet_models/model-559.pt test.tif
