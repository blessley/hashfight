#!/bin/bash

#hashfight build dir
BIN="/home/users/blessley/build-hash-fight"

#dir with the binary key/val files
data_DIR="/home/users/blessley/hashing-data"

#run the application:
${BIN}/Hashing_CUDA 900000000 1.03 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 950000000 1.03 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1000000000 1.03 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1050000000 1.03 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1250000000 1.03 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1450000000 1.03 0 10 0 ${data_DIR}

${BIN}/Hashing_CUDA 900000000 1.03 5 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 950000000 1.03 5 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1000000000 1.03 5 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1050000000 1.03 5 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1250000000 1.03 5 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 1450000000 1.03 5 10 0 ${data_DIR}

${BIN}/Hashing_CUDA 50000000 2.0 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 250000000 2.0 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 500000000 2.0 0 10 0 ${data_DIR}
${BIN}/Hashing_CUDA 900000000 2.0 0 10 0 ${data_DIR}

