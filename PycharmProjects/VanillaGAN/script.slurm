#!/bin/bash
#SBATCH --job-name=gan
#SBATCH -o  gan.out -e  gan.err
#NC="-c 1"
#NP="-n 1"
#REQUEST="--mem=4000"
#SBATCH --gres=gpu:K20Xm:1
#SBATCH -t 300

module load gcc/latest
module load nvidia/7.5
module load cudnn/7.5-v5

python -m train.training_functions
##python run_model.py resnet_models/model-559.pt test.tif
