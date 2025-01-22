#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv

cd /global/homes/y/ycryu/wip/bert-finetuning
python main.py --train --epochs 10 --model-save-path /pscratch/sd/y/ycryu/bert-checkpoints/