cmake_minimum_required(VERSION 3.6)
#set (CMAKE_CXX_STANDARD 11)
set(PROJECT_NAME "hashing-benchmark")

project(${PROJECT_NAME})
#add_executable(${PROJECT_NAME} connComps.cxx)

#Find the CUDPP package
set(cudpp_DIR "/home/users/blessley/build-cudpp/lib")
find_package(cudpp CONFIG REQUIRED)
include(/home/users/blessley/build-cudpp/lib/cudpp-targets.cmake)

#Find the VTK-m package
set(VTKm_DIR "/home/users/blessley/test-apurna/lib/cmake/vtkm-1.2")
find_package(VTKm REQUIRED QUIET)

add_executable(Hashing_SERIAL HashingBenchmark.cxx)
target_include_directories(Hashing_SERIAL PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "/home/users/blessley/cudpp/src/cudpp" "/home/users/blessley/cudpp/src/cudpp_hash")
target_link_libraries(Hashing_SERIAL PRIVATE vtkm_cont cudpp_hash cudpp)
target_compile_definitions(Hashing_SERIAL PRIVATE
                           "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_SERIAL")


if(TARGET vtkm::tbb)
  add_executable(Hashing_TBB HashingBenchmarkTBB.cxx)
  target_include_directories(Hashing_TBB PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "/home/users/blessley/cudpp/src/cudpp" "/home/users/blessley/cudpp/src/cudpp_hash")
  target_compile_definitions(Hashing_TBB PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_TBB")
  target_link_libraries(Hashing_TBB PRIVATE vtkm_cont cudpp_hash cudpp)
endif()


if(TARGET vtkm::cuda)
  #vtkm_compile_as_cuda(cudaSource HashingBenchmark.cu)
  add_executable(Hashing_CUDA HashingBenchmark.cu) 
  target_include_directories(Hashing_CUDA PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "/home/users/blessley/cudpp/src/cudpp" "/home/users/blessley/cudpp/src/cudpp_hash")
  target_compile_definitions(Hashing_CUDA PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_CUDA")
  target_link_libraries(Hashing_CUDA PRIVATE vtkm_cont cudpp_hash cudpp)
endif()



