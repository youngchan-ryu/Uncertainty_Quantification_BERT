#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv

cd /global/homes/y/ycryu/wip/UQ/bert
python main.py --evaluate --model-save-path /global/homes/y/ycryu/wip/UQ/bert/checkpoints/checkpoint_20250110_084205_epoch10.pt --uncertainty --dropout-rate 0.5 > /global/homes/y/ycryu/wip/UQ/bert/test_result/epoch10-50-2.log &
python main.py --evaluate --model-save-path /global/homes/y/ycryu/wip/UQ/bert/checkpoints/checkpoint_20250109_232807_epoch1.pt --uncertainty --dropout-rate 0.5 > /global/homes/y/ycryu/wip/UQ/bert/test_result/epoch1-50-3.log &
wait