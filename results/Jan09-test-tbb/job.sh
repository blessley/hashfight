#!/bin/bash -l
#SBATCH -J CPU-Jan09
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --mem=370G
#SBATCH -t 24:00:00
#SBATCH -A nvst0001
#SBATCH -p dav
#SBATCH -C skylake

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

module remove intel openmpi
module load gnu/6.4.0 cmake 

/glade/work/shaomeng/XeonGold/hashfight/results/Jan09-test-tbb/run-all-cpu-Jan09.sh;

### #SBATCH --ntasks-per-node=72