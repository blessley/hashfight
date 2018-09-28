cmake_minimum_required(VERSION 3.6)
#set (CMAKE_CXX_STANDARD 11)
set(PROJECT_NAME "cudpp-cuckoo")

project(${PROJECT_NAME})

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_35,code=sm_35
    )

#Find the CUDPP package
set(cudpp_DIR "/home/users/blessley/build-cudpp/lib")
find_package(cudpp CONFIG REQUIRED)
include(/home/users/blessley/build-cudpp/lib/cudpp-targets.cmake)

add_executable(CuckooHash CuckooBenchmark.cxx)
target_include_directories(CuckooHash PRIVATE ${CUDA_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "/home/users/blessley/cudpp/src/cudpp" "/home/users/blessley/cudpp/src/cudpp_hash")
target_link_libraries(CuckooHash PRIVATE ${CUDA_LIBRARIES} cudpp_hash cudpp)


