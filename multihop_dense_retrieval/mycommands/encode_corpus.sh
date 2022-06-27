#!/bin/bash
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here

eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate MDR
​
export TQDM_DISABLE=1
export OMP_NUM_THREADS=1

# python -m torch.distributed.launch --nproc_per_node=2 scripts/encode_corpus.py \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file /projects/tir3/users/nnishika/stqa_corpus.json \
    --init_checkpoint /home/nnishika/multihop_dense_retrieval/models/doc_encoder.pt \
    --embed_save_path /home/nnishika/StqaIndex \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20
    # --local_rank 0
