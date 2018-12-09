#!/bin/bash
#SBATCH --job-name=NN-FINN
#SBATCH -o  forest.out -e  forest.err
#NC="-c 1"
#NP="-n 1"
#REQUEST="--mem=4000"
#SBATCH --gres=gpu:K20Xm:1
#SBATCH -t 800

module load gcc/latest
module load nvidia/7.5
module load cudnn/7.5-v5

python train.py --function train_net --data_path tif/ --save_dir vgg5 --batch_size 256 --lr 0.0022 --log_after 5 --cuda 1 --device 0
##python train.py --function eval_net --data_path tif/ --save_dir vgg5 --batch_size 1 --lr 0.0022 --log_after 1 --cuda 1 --device 0 --pretrained_model vgg5/model-234.pt
##python run_model.py resnet_models/model-559.pt test.tif
