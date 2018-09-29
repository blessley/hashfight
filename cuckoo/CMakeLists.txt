cmake_minimum_required(VERSION 3.6)
set (CMAKE_CXX_STANDARD 11)
set(PROJECT_NAME "cudpp-cuckoo")

set(CUDPP_SRC "/glade/work/shaomeng/cudpp")
set(cudpp_DIR "/glade/work/shaomeng/v100/cudpp/lib")

project(${PROJECT_NAME})

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_70,code=sm_70
    )

#Find the CUDPP package
find_package(cudpp CONFIG REQUIRED)
include( ${cudpp_DIR}/cudpp-targets.cmake )

add_executable(CuckooHash CuckooBenchmark.cxx)
target_include_directories(CuckooHash PRIVATE ${CUDA_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} ${CUDPP_SRC}/src/cudpp ${CUDPP_SRC}/src/cudpp_hash )
target_link_libraries(CuckooHash PRIVATE ${CUDA_LIBRARIES} cudpp_hash cudpp)


