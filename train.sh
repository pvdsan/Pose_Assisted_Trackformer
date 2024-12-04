#!/bin/bash
#SBATCH -p qTRDGPUH
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -e error%A.err 
#SBATCH -o out%A.out
#SBATCH -A trends396s109
#SBATCH -J Working_Memory_Prediction
#SBATCH --oversubscribe
#SBATCH --mail-user=pvdsan@gmail.com
#SBATCH --exclude=arctrdagn047

# a small delay at the start often helps
sleep 2s 

#activate the environment
source /home/users/sdeshpande8/anaconda3/bin/activate trackformer

# CD into your directory
cd /data/users4/sdeshpande8/DL_Assignements/trackformer

# run the batch script
torchrun --nproc_per_node=4 src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/mot17_deformable_multi_frame

# a delay at the end is also good practice
sleep 10s
