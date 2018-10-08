#!/bin/bash -l
#SBATCH -J GPU-Oct07
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -A nvst0001
#SBATCH -p dav
#SBATCH -e job_name.err.%J
#SBATCH -o job_name.out.%J
#SBATCH --gres=gpu:v100:1

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module remove intel openmpi
module load gnu/6.4.0 cuda cmake 

srun /glade/work/shaomeng/v100/hashfight/results/Oct07/run-all-gpu-Oct-07.sh
