cmake_minimum_required(VERSION 3.6)
set ( CMAKE_CXX_STANDARD 11)
set( PROJECT_NAME "hashing-benchmark")

project(${PROJECT_NAME})

#SET THESE RELATIVE TO YOUR SYSTEM:
set( VTKm_DIR "/glade/work/shaomeng/XeonGold/vtkm-build/lib/cmake/vtkm-1.2" )

# Set TBB directories
set( TBB_INCLUDE_DIR "/glade/u/home/shaomeng/Install/tbb2018_20180618oss/include" )
set( TBB_LIBRARY_DEBUG  "/glade/u/home/shaomeng/Install/tbb2018_20180618oss/lib/intel64/gcc4.7/libtbb_debug.so" )
set( TBB_LIBRARY_RELEASE  "/glade/u/home/shaomeng/Install/tbb2018_20180618oss/lib/intel64/gcc4.7/libtbb.so" )

set( SOURCES random_numbers.cpp mt19937ar.cpp)

#Finds the VTK-m package
find_package(VTKm REQUIRED QUIET)

add_executable(Hashing_SERIAL HashingBenchmark.cxx ${SOURCES})
target_include_directories(Hashing_SERIAL PRIVATE ${VTKm_INCLUDE_DIRS} include)
target_link_libraries(Hashing_SERIAL PRIVATE vtkm_cont)
target_compile_definitions(Hashing_SERIAL PRIVATE
                           "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_SERIAL")

if(TARGET vtkm::tbb)
  add_executable(Hashing_TBB HashingBenchmarkTBB.cxx ${SOURCES})
  target_include_directories(Hashing_TBB PRIVATE ${VTKm_INCLUDE_DIRS} include)
  target_compile_definitions(Hashing_TBB PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_TBB")
  target_link_libraries(Hashing_TBB PRIVATE vtkm_cont)
endif()


if(TARGET vtkm::cuda)
  add_executable(Hashing_CUDA HashingBenchmark.cu ${SOURCES}) 
  target_include_directories(Hashing_CUDA PRIVATE ${VTKm_INCLUDE_DIRS} include)
  target_compile_definitions(Hashing_CUDA PRIVATE
                             "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_CUDA")
  target_link_libraries(Hashing_CUDA PRIVATE vtkm_cont)
endif()



