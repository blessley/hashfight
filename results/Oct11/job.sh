#!/bin/bash -l
#SBATCH -J CPU-Oct11
#SBATCH -n 72
#SBATCH --ntasks-per-node=72
#SBATCH --mem=370G
#SBATCH -t 24:00:00
#SBATCH -A nvst0001
#SBATCH -p dav
#SBATCH -C skylake

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module remove intel openmpi
module load gnu/6.4.0 cmake 

/glade/work/shaomeng/XeonGold/hashfight/results/Oct11/run-all-cpu-Oct11.sh
