#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=youngchanryu@snu.ac.kr

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv

export SLURM_CPU_BIND="cores"
srun ./training.sh