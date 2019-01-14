#!/bin/bash

#SET THESE RELATIVE TO YOUR SYSTEM:
#hashfight_SRC="/glade/work/shaomeng/XeonGold/hashfight/hashfight"
hashfight_BUILD="/home/users/blessley/build-hash-fight"
thrust_BUILD="/home/users/blessley/HashFight/thrust-tbb"
unordered_map_BUILD="/home/users/blessley/HashFight/unordered-map"
hashing_DATA="/home/users/blessley/hashing-data" #HashFight data dir
#timings_OUT_DIR="/glade/work/shaomeng/XeonGold/hashfight/results/Oct05-test-tbb" #dir for output timing files


k='1450000000'
l='1.03'
f='0'
failure_trials='10'
counter='0'

#SET THE NUMBER OF TBB THREADS HERE:
NUM_TBB_THREADS='32'

#Test the number of TBB threads being used for each algorithm
${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
env TBB_VERSION=1 ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS
env TBB_VERSION=1 ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS

