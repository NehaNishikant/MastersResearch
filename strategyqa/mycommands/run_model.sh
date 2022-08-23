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
conda activate stqa
​
export TQDM_DISABLE=1
# code

python -m src.models.iterative.run_model \
    -g 0 \ 
    --qa-model-path /projects/tir3/users/nnishika/3_STAR_IR-Q.tar.gz \ 
    --paragraphs-source IR-Q \
    --data data/strategyqa/train.json #\ 
#    --output-predictions-file stqaout/run_model_out.jsonl

