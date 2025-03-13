#!/bin/bash
#SBATCH --job-name=dyna1
#SBATCH --output=jobs/dyna1.%j.out
#SBATCH --error=jobs/dyna1.%j.err
#SBATCH --partition=preempt
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --constraint='L40S'
#SBATCH --exclude='babel-14-25,babel-11-5,babel-1-23,babel-13-13'


export NCCL_P2P_DISABLE=1

source /usr/share/Modules/init/bash
module load cuda-12.4

export PATH=/home/sroutra2/miniconda3/envs/lq/bin:$PATH
# export LD_LIBRARY_PATH=/home/sroutra2/miniconda3/envs/lq/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
accelerate launch --num_processes=4 --mixed_precision=bf16 --main_process_port=27562 train_dynamics1.py
