# #!/bin/bash
# #SBATCH --job-name=dyna1_ft
# #SBATCH --output=jobs/dyna1_ft.%j.out
# #SBATCH --error=jobs/dyna1_ft.%j.err
# #SBATCH --partition=general
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=100G
# #SBATCH --time=2-00:00:00
# #SBATCH --constraint='6000Ada|L40S|A6000'
# #exclude='babel-14-25,babel-11-5,babel-1-23,babel-13-13'


# export NCCL_P2P_DISABLE=1

# source /usr/share/Modules/init/bash
# module load cuda-12.4

# export PATH=/home/sroutra2/miniconda3/envs/lq/bin:$PATH
# # export LD_LIBRARY_PATH=/home/sroutra2/miniconda3/envs/lq/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export WANDB_USER_NAME="saintlyk1d"
export WANDB_API_KEY="eb8c78ce366d7ad5e6791379dab1c1a0157fa60b"
wandb login $WANDB_API_KEY
accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port=27562 ft_dynamics1.py
