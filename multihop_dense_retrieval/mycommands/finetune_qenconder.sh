#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here

source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate MDR
​
export TQDM_DISABLE=1


# -- mem = 256 --gres=gpu:1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_momentum.py \
    --do_train \
    --prefix 0 \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file /home/nnishika/stqaout/finetune_mdr/mdr_trainfile.jsonl \
    --predict_file /home/nnishika/stqaout/finetune_mdr/mdr_evalfile.jsonl \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --momentum \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init-retriever models/q_encoder.pt
