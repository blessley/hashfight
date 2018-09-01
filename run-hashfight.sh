#!/bin/bash -l

BIN="/home/users/blessley/build-hash-fight"


#run the application:
nvprof --csv -u ms --print-gpu-summary --log-file timings ${BIN}/Hashing_CUDA $1 $2 $3 $4 $5; cat timings | tee >(grep "CuckooHashing" | tee >(grep "retrieve" | awk -F"," '{print "CuckooHash Query:", $2}') >(grep -v "retrieve" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "CuckooHash Insert:", s}')) >(grep -v "CuckooHashing\|memset\|memcpy" | tee >(grep "ProbeForKey\|ConstantFunctor<unsigned int>" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "HashFight Query:", s}') >(grep -v "ProbeForKey\|ConstantFunctor<unsigned int>" | awk -F"," '{print $2}' | awk '{s+=$1} END {print "HashFight Insert:", s}')) | grep "Insert:\|Query:" | tail -4 | sed 's/.*: //' | paste -sd ','


