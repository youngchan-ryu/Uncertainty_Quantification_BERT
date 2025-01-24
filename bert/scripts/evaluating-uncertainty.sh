#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate myenv

cd /global/homes/y/ycryu/wip/UQ/bert
python main.py --evaluate --model-save-path /global/homes/y/ycryu/wip/UQ/bert/checkpoints/checkpoint_20250110_084205_epoch10.pt --uncertainty --dropout-rate 0.5 > /global/homes/y/ycryu/wip/UQ/bert/test_result/epoch10-50-2.log &
python main.py --evaluate --model-save-path /global/homes/y/ycryu/wip/UQ/bert/checkpoints/checkpoint_20250109_232807_epoch1.pt --uncertainty --dropout-rate 0.5 > /global/homes/y/ycryu/wip/UQ/bert/test_result/epoch1-50-3.log &
wait

# python -m pdb main.py --evaluate --model-save-path /scratch/connectome/ycryu/UQ/bert/bert-checkpoints/checkpoint_20250123_061001_epoch10.pt --uncertainty --dropout-rate 0.5 > /scratch/connectome/ycryu/UQ/bert/test_result/epoch10-50-1.log
# python main.py --evaluate --model-save-path /scratch/connectome/ycryu/UQ/bert/bert-checkpoints/checkpoint_20250123_111101_epoch20.pt --uncertainty --dropout-rate 0.5 > /scratch/connectome/ycryu/UQ/bert/test_result/epoch20-50-1.log