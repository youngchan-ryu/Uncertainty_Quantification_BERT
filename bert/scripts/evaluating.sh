#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv

cd /global/homes/y/ycryu/wip/bert-finetuning
python main.py --evaluate --model-save-path /global/homes/y/ycryu/wip/UQ/bert/checkpoints/checkpoint_20250110_084205_epoch10.pt