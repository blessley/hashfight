#!/bin/bash

#SET THESE RELATIVE TO YOUR SYSTEM:
hashfight_SRC="/glade/work/shaomeng/XeonGold/hashfight/hashfight"
hashfight_BUILD="/glade/work/shaomeng/XeonGold/hashfight/hashfight/build"
thrust_BUILD="/glade/work/shaomeng/XeonGold/hashfight/thrust-tbb"
unordered_map_BUILD="/glade/work/shaomeng/XeonGold/hashfight/unordered-map"
hashing_DATA="/glade/scratch/shaomeng/hashing-data" #HashFight data dir
timings_OUT_DIR="/glade/work/shaomeng/XeonGold/hashfight/results/Jan15" #dir for output timing files

factors='1.03'
failure='0'
failure_trials=10
all_trials=10

NUM_TBB_THREADS=72


#: <<'COMMENT'
#run through all the sizes for a factor-failure configuration:
filename_config_times="${timings_OUT_DIR}/${factors}-${failure}-${failure_trials}"
touch $filename_config_times
filename_temp_results="${timings_OUT_DIR}/temp"
touch $filename_temp_results
for model in {1..3}; do #models to test
  for l in $factors; do #load factors
    for f in $failure; do #failure rates
      filename_trial_times="${timings_OUT_DIR}/${model}-${l}-${f}-${failure_trials}"
      touch $filename_trial_times
      : > $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        : > $filename_temp_results
        for k in {50000000..1450000000..50000000}; do #num key-val pairs
        #for k in $sizes; do
          case $model in
          1) ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          2) ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          3) ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          esac
        done
        paste -d, -s $filename_temp_results >> $filename_trial_times
        ((counter++))
      done
      cat $filename_trial_times | \
      awk -F',' '{for (i=1;i<=NF;i++){a[i]+=$i;}} END {printf "%.4f", a[1]/NR; for (i=2;i<=NF;i++){printf ",%.4f", a[i]/NR};}' \
      >> $filename_config_times 2>&1
      echo "" >> $filename_config_times 2>&1
    done
  done
done
#COMMENT

factors='1.03 1.10 1.15 1.25 1.40 1.50 1.60 1.75 1.90 2.0'
sizes='1450000000'
failure='0'
#: <<'COMMENT'
#run through all the factors for a size-failure configuration:
filename_config_times="${timings_OUT_DIR}/${sizes}-${failure}-${failure_trials}"
touch $filename_config_times
filename_temp_results="${timings_OUT_DIR}/temp"
touch $filename_temp_results
for model in {1..3}; do #models to test
  for k in $sizes; do #load factors
    for f in $failure; do #failure rates
      filename_trial_times="${timings_OUT_DIR}/${model}-${k}-${f}-${failure_trials}"
      touch $filename_trial_times
      : > $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        : > $filename_temp_results
        #for k in {50000000..1450000000..50000000}; do #num key-val pairs
        for l in $factors; do
          case $model in
          1) ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          2) ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          3) ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          esac
        done
        paste -d, -s $filename_temp_results >> $filename_trial_times
        ((counter++))
      done
      cat $filename_trial_times | \
      awk -F',' '{for (i=1;i<=NF;i++){a[i]+=$i;}} END {printf "%.4f", a[1]/NR; for (i=2;i<=NF;i++){printf ",%.4f", a[i]/NR};}' \
      >> $filename_config_times 2>&1
      echo "" >> $filename_config_times 2>&1
    done
  done
done
#COMMENT

sizes='1450000000'
factors='1.03'
#: <<'COMMENT'
#run through all the failures for a size-factor configuration:
filename_config_times="${timings_OUT_DIR}/${sizes}-${factors}"
touch $filename_config_times
filename_temp_results="${timings_OUT_DIR}/temp"
touch $filename_temp_results
for model in {1..3}; do #models to test
  for l in $factors; do #load factors
    for k in $sizes; do #failure rates
      filename_trial_times="${timings_OUT_DIR}/${model}-${k}-${l}"
      touch $filename_trial_times
      : > $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        : > $filename_temp_results
        #for k in {50000000..1450000000..50000000}; do #num key-val pairs
        for f in {0..9}; do
          case $model in
          1) ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          2) ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          3) ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS >> $filename_temp_results 2>&1
             ;;
          esac
        done
        paste -d, -s $filename_temp_results >> $filename_trial_times
        ((counter++))
      done
      cat $filename_trial_times | \
      awk -F',' '{for (i=1;i<=NF;i++){a[i]+=$i;}} END {printf "%.4f", a[1]/NR; for (i=2;i<=NF;i++){printf ",%.4f", a[i]/NR};}' \
      >> $filename_config_times 2>&1
      echo "" >> $filename_config_times 2>&1
    done
  done
done
#COMMENT


