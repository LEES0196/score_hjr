#!/bin/bash

#SBATCH --job-name=sde
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

module load Miniforge3
module load CUDA/11.7.0
conda activate otflow
python trainMnistOTflow.py \
	--autoenc experiments/cnf/large/2025_05_09_14_03_44_autoenc_checkpt.pth \
	--batch_size 128 \
	--nt 128 \
	--alph '1.0, 0.0, 0.0'\
	--convex True \
	--wandb_log True


python trainMnistOTflow.py --autoenc experiments/cnf/large/2025_05_09_14_03_44_autoenc_checkpt.pth --batch_size 128 --nt 128 --alph '1.0,0.0,0.0' --lr 0.001 --convex True
