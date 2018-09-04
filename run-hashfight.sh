#!/bin/bash

#hashfight build dir
BIN="/home/users/blessley/build-hash-fight"

#dir with the binary key/val files
data_DIR="/home/users/blessley/hashing-data"

#run the application:
nvprof --csv -u ms --print-gpu-summary --log-file timings ${BIN}/Hashing_CUDA $1 $2 $3 $4 $5 ${data_DIR}; cat timings | tee >(grep "CuckooHashing" | tee >(grep "retrieve" | awk -F"," '{print "CuckooHash Query:", $2}') >(grep -v "retrieve" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "CuckooHash Insert:", s}')) >(grep -v "CuckooHashing\|memset\|memcpy" | tee >(grep "ProbeForKey\|ConstantFunctor<unsigned int>" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "HashFight Query:", s}') >(grep -v "ProbeForKey\|ConstantFunctor<unsigned int>" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "HashFight Insert:", s}')) | grep "Insert:\|Query:" | tail -4 | sed 's/.*: //' | paste -sd ','


