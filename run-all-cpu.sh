#!/bin/bash

#SET THESE RELATIVE TO YOUR SYSTEM:
hashfight_SRC="/home/users/blessley/HashFight" #HashFight source dir
hashfight_BUILD="/home/users/blessley/build-hash-fight" #HashFight build dir
thrust_BUILD="/home/users/blessley/HashFight/thrust-tbb" #Sort+Search thrust tbb benchmark dir
unordered_map_BUILD="/home/users/blessley/HashFight/unordered-map" #tbb benchmark build dir
hashing_DATA="/home/users/samuelli/Datasets/hashing-data" #data dir
timings_OUT_DIR="/home/users/blessley/hashing-timings/cpu-10-01" #dir for output timing files

factors='1.03'
failure='0'
failure_trials=10
all_trials=10

: <<'COMMENT'
#run through all the sizes for a factor-failure configuration:
for l in $factors; do #load factors
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${l}-${f}-${failure_trials}"
    touch $filename_config_times
    for k in {50000000..500000000..50000000}; do #num key-val pairs
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT


factors='1.50'
failure='0'
#: <<'COMMENT'
#run through all the sizes for a factor-failure configuration:
for l in $factors; do #load factors
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${l}-${f}-${failure_trials}"
    touch $filename_config_times
    for k in {50000000..500000000..50000000}; do #num key-val pairs
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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


factors='2.0'
failure='0'
: <<'COMMENT'
#run through all the sizes for a factor-failure configuration:
for l in $factors; do #load factors
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${l}-${f}-${failure_trials}"
    touch $filename_config_times
    for k in {50000000..500000000..50000000}; do #num key-val pairs
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT

sizes='500000000'
factors='1.03'
: <<'COMMENT'
#run through all the failures for a size-factor configuration:
for k in $sizes; do #num key-val pairs
  for l in $factors; do #load factor
    filename_config_times="${timings_OUT_DIR}/${k}-${l}"
    touch $filename_config_times
    for f in {0..9}; do #failure rates
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT


sizes='500000000'
factors='2.0'
: <<'COMMENT'
#run through all the failures for a size-factor configuration:
for k in $sizes; do #num key-val pairs
  for l in $factors; do #load factor
    filename_config_times="${timings_OUT_DIR}/${k}-${l}"
    touch $filename_config_times
    for f in {0..9}; do #failure rates
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT

factors='1.03 1.10 1.15 1.25 1.40 1.50 1.60 1.75 1.90 2.0'
sizes='500000000'
failure='0'
: <<'COMMENT'
#run through all the factors for a size-failure configuration:
for k in $sizes; do #num key-val pairs
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${k}-${f}-${failure_trials}"
    touch $filename_config_times
    #for k in {25000000..500000000..25000000}; do #num key-val pairs
    for l in $factors; do #load factors
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT

factors='1.03 1.10 1.15 1.25 1.40 1.50 1.60 1.75 1.90 2.0'
sizes='500000000'
failure='5'
: <<'COMMENT'
#run through all the factors for a size-failure configuration:
for k in $sizes; do #num key-val pairs
  for f in $failure; do #failure rates
    filename_config_times="${timings_OUT_DIR}/${k}-${f}-${failure_trials}"
    touch $filename_config_times
    #for k in {25000000..500000000..25000000}; do #num key-val pairs
    for l in $factors; do #load factors
      filename_trial_times="${timings_OUT_DIR}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times 
      filename_temp_results="${timings_OUT_DIR}/temp"
      touch $filename_temp_results
      counter=0
      while [ $counter -lt $all_trials ]; do 
        ${hashfight_BUILD}/Hashing_TBB $k $l $f $failure_trials $counter $hashing_DATA > $filename_temp_results 2>&1
        ${unordered_map_BUILD}/UnorderedMap $k $l $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
        ${thrust_BUILD}/SortSearch $k $f $failure_trials $counter $hashing_DATA >> $filename_temp_results 2>&1
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
COMMENT


