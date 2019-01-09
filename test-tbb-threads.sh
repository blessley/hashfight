#!/bin/bash

#SET THESE RELATIVE TO YOUR SYSTEM:
hashfight_SRC="/glade/work/shaomeng/XeonGold/hashfight/hashfight"
hashfight_BUILD="/glade/work/shaomeng/XeonGold/hashfight/hashfight/build"
thrust_BUILD="/glade/work/shaomeng/XeonGold/hashfight/thrust-tbb"
unordered_map_BUILD="/glade/work/shaomeng/XeonGold/hashfight/unordered-map"
hashing_DATA="/glade/scratch/shaomeng/hashing-data" #HashFight data dir
timings_OUT_DIR="/glade/work/shaomeng/XeonGold/hashfight/results/Jan06-test-tbb" #dir for output timing files


k='1450000000'
l='2.0'
f='0'
failure_trials='10'
counter='0'

#SET THE NUMBER OF TBB THREADS HERE:
# (> cores = hyperthreading)
NUM_TBB_THREADS='72'

#Test the number of TBB threads being used for each algorithm
${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
#${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS

NUM_TBB_THREADS='72'

${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
#${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
