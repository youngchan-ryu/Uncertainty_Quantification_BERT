#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv
# conda activate pyenv

cd /global/homes/y/ycryu/wip/bert-finetuning
# python main.py --train --epochs 10 --model-save-path /pscratch/sd/y/ycryu/bert-checkpoints/
python main.py --train --epochs 20 --model-save-path /scratch/connectome/ycryu/UQ/bert/bert-checkpoints/