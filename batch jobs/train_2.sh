#!/bin/bash

#SBATCH -J graph
#SBATCH -t 12:00:00
#SBATCH --mail-user=csauac@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -p normal
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --account=mscbdt2024
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

cd msbd5008
conda activate graph
module load slurm 'nvhpc-hpcx-cuda12/23.11'
python graph_train.py --graph_idx 2
