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
python3 run_scripts/evaluate.py --model /projects/tir3/users/nnishika/6_STAR_ORA-P.tar.gz --data data/strategyqa/dev.json -g 0
