#!/usr/bin/bash
#SBATCH --mem=16gb                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                # Number of gpu
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=./log/%j.log   # Standard output and error log, the program output will be here
​
# you can always have this
eval "$(conda shell.bash hook)"
# you environment
source /home/nnishika/miniconda3/etc/profile.d/conda.sh
conda activate base
​
export TQDM_DISABLE=1
# code

git add -A strategyqa
git add -A multihop_dense_retrieval/scripts
git add -A multihop_dense_retrieval/mdr
git add -A multihop_dense_retrieval/mycommands
git add -A DPR/dpr
git add -A DPR/mycommands
git add -A DPR/conf
git add -A DPR/dense_retriever.py
git add -A DPR/download_wiki.py
git add -A DPR/generate_dense_embeddings.py
git add -A DPR/setup.py
git add -A DPR/train_dense_encoder.py
git add -A DPR/train_extractive_reader.py
git add -A data_utils.py
git add -A data_utils.sh
git add -A git_add.sh
git add -A requirements_base.txt
git add -A requirements_mdr.txt
git add -A requirements_dpr.txt
git add -A requirements_stqa.txt
git add -A test/readme.txt
git add -A stqaout/
git add -A mdrout/
git add -A dprout/
git add -A out/
