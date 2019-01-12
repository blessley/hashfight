#!/bin/bash

#SET THESE RELATIVE TO YOUR SYSTEM:
hashfight_SRC="/glade/work/shaomeng/XeonGold/hashfight/hashfight"
hashfight_BUILD="/glade/work/shaomeng/XeonGold/hashfight/hashfight/build"
thrust_BUILD="/glade/work/shaomeng/XeonGold/hashfight/thrust-tbb"
unordered_map_BUILD="/glade/work/shaomeng/XeonGold/hashfight/unordered-map"
hashing_DATA="/glade/scratch/shaomeng/hashing-data" #HashFight data dir
timings_OUT_DIR="/glade/work/shaomeng/XeonGold/hashfight/results/Jan12" #dir for output timing files

factors='1.03'
failure='9'
failure_trials=10
all_trials=5

NUM_TBB_THREADS=72

#: <<'COMMENT'
#run through all the sizes for a factor-failure configuration:
for l in $factors; do #load factors
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${l}-${f}-${failure_trials}"
    touch $filename_config_times
    for k in {500000000..500000000..50000000}; do #num key-val pairs
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm -f $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS  > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS  >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA $NUM_TBB_THREADS  >> $filename_temp_results 2>&1
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



