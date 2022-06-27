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

# conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
nvidia-smi # --list-gpus
nvcc --version
python test.py 
