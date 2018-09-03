cmake_minimum_required(VERSION 3.6)
#set (CMAKE_CXX_STANDARD 11)
set(PROJECT_NAME "hashing-benchmark")

project(${PROJECT_NAME})


#SET THESE RELATIVE TO YOUR SYSTEM:
set(cudpp_SRC "/home/users/blessley/cudpp") #cudpp source dir
set(cudpp_DIR "/home/users/blessley/build-cudpp/lib") #cudpp build dir
set(VTKm_DIR "/home/users/blessley/test-apurna/lib/cmake/vtkm-1.2") #vtkm build dir


#Finds the CUDPP package
find_package(cudpp CONFIG REQUIRED)
include(${cudpp_DIR}/cudpp-targets.cmake)

#Finds the VTK-m package
find_package(VTKm REQUIRED QUIET)

add_executable(Hashing_SERIAL HashingBenchmark.cxx)
target_include_directories(Hashing_SERIAL PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "${cudpp_SRC}/src/cudpp" "${cudpp_SRC}/src/cudpp_hash")
target_link_libraries(Hashing_SERIAL PRIVATE vtkm_cont cudpp_hash cudpp)
target_compile_definitions(Hashing_SERIAL PRIVATE
                           "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_SERIAL")

if(TARGET vtkm::tbb)
  add_executable(Hashing_TBB HashingBenchmarkTBB.cxx)
  target_include_directories(Hashing_TBB PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "${cudpp_SRC}/src/cudpp" "${cudpp_SRC}/src/cudpp_hash")
  target_compile_definitions(Hashing_TBB PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_TBB")
  target_link_libraries(Hashing_TBB PRIVATE vtkm_cont cudpp_hash cudpp)
endif()


if(TARGET vtkm::cuda)
  add_executable(Hashing_CUDA HashingBenchmark.cu) 
  target_include_directories(Hashing_CUDA PRIVATE ${VTKm_INCLUDE_DIRS} ${cudpp_INCLUDE_DIRS} "${cudpp_SRC}/src/cudpp" "${cudpp_SRC}/src/cudpp_hash")
  target_compile_definitions(Hashing_CUDA PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_CUDA")
  target_link_libraries(Hashing_CUDA PRIVATE vtkm_cont cudpp_hash cudpp)
endif()



