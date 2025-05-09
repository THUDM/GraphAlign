# This script is used to load the necessary modules for running a job on a cluster.
cd msbd5008
conda activate graph
module load slurm 'nvhpc-hpcx-cuda12/23.11'


#previous version
# 2.4.0+cu121
#12.1
#module load slurm 'cuda11.8/toolkit/11.8.0'
