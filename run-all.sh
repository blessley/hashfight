#!/bin/bash -l

HOME="/home/users/blessley"
BIN="${HOME}/HashFight"
DATA="${HOME}/hashing-data"
TIMES="${HOME}/hashing-timings"

#sizes='500000000 1048576 134217728
#factors='1.03 2.00 1.50 1.25 1.75 1.10 1.15 1.35 1.90'
factors='1.03'
#factors='2.00'
sizes='325000000'
failure='5'
failure_trials=10
all_trials=1

#: <<'COMMENT'
#run through all the sizes for each factor-failure configuration:
for l in $factors; do #load factors
  for f in $failure; do #failure rates
    filename_config_times="${TIMES}/${l}-${f}-${failure_trials}"
    touch $filename_config_times
    for k in {25000000..500000000..25000000}; do #num key-val pairs
    #for k in $sizes; do #num key-val pairs
      filename_trial_times="${TIMES}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${BIN}/run-hashfight.sh $k $l $f $failure_trials $counter >> $filename_trial_times 2>&1
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


: <<'COMMENT'
#run through all the factors for each size-failure configuration:
for k in $sizes; do #num key-val pairs
  for f in $failure; do #failure rates
    filename_config_times="${TIMES}/${k}-${f}-${failure_trials}"
    touch $filename_config_times
    #for k in {25000000..500000000..25000000}; do #num key-val pairs
    for l in $factors; do #load factors
      filename_trial_times="${TIMES}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${BIN}/run-hashfight.sh $k $l $f $failure_trials $counter >> $filename_trial_times 2>&1
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

: <<'COMMENT'
#run through all the failures for each size-factor configuration:
for k in $sizes; do #num key-val pairs
  for l in $factors; do #load factor
    filename_config_times="${TIMES}/${k}-${l}"
    touch $filename_config_times
    #for k in {25000000..500000000..25000000}; do #num key-val pairs
    for f in {0..9}; do #failure rates
      filename_trial_times="${TIMES}/${k}-${l}-${f}-${failure_trials}"
      if [ -f "$filename_trial_times" ]; then
        rm $filename_trial_times
      fi
      touch $filename_trial_times
      counter=0
      while [ $counter -lt $all_trials ]; do
        ${BIN}/run-hashfight.sh $k $l $f $failure_trials $counter >> $filename_trial_times 2>&1
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

