#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <iostream>
#include <string>
#include <fstream>

template <typename T>
void load_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ifstream ifile(filename, std::ios::binary);
    ifile.read((char*) data, sizeof(T)*length);
    ifile.close();
}


int main(int argc, char** argv)
{


  const std::string data_dir(argv[5]);
  float sortTime = 0, searchTime = 0;
  cudaEvent_t start, stop;

  //std::cout << "========================Thrust Sort+BinarySearch"
    //        << "==============================\n";


  unsigned int kInputSize = (unsigned int)std::atoi(argv[1]);
  unsigned int* input_keys = new unsigned int[kInputSize];
  unsigned int* input_vals = new unsigned int[kInputSize];
  unsigned int* query_keys = new unsigned int[kInputSize];
  
  //std::cout << "Loading binary of input keys...\n";
  load_binary(input_keys, kInputSize, data_dir + "/inputKeys-" + std::string(argv[1]) + "-" + std::string(argv[4])); 

  //std::cout << "Loading binary of input vals...\n";
  load_binary(input_vals, kInputSize, data_dir + "/inputVals-" + std::string(argv[1]) + "-" + std::string(argv[4])); 

  //std::cout << "Loading binary of query keys...\n";
  load_binary(query_keys, kInputSize, data_dir + "/queryKeys-" + std::string(argv[1]) + "-" + std::string(argv[2]) + "-" + std::string(argv[3]) + "-" + std::string(argv[4])); 

 
  thrust::device_vector<unsigned int> keys_d(input_keys,
					     input_keys + kInputSize);

  thrust::device_vector<unsigned int> vals_d(input_vals,
					     input_vals + kInputSize);

  //std::cout << "Sorting pairs...\n";
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  thrust::sort_by_key(keys_d.begin(), keys_d.end(), vals_d.begin());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&sortTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //std::cout << "Sort: elapsed : " << sortTime/1000 << "\n";
  
  #if 1
  thrust::device_vector<unsigned int> query_keys_d(query_keys,
					           query_keys + kInputSize);
  
  thrust::device_vector<unsigned int> search_result(kInputSize);

  //std::cout << "Searching for pairs...\n";
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  thrust::binary_search(keys_d.begin(), keys_d.end(), 
                        query_keys_d.begin(), query_keys_d.end(),
                        search_result.begin());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&searchTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  delete [] query_keys;
  #endif

  //std::cout << "Search: elapsed : " << searchTime/1000 << "\n";
  std::cout << sortTime/1000 << "\n";
  std::cout << searchTime/1000 << "\n";

  delete [] input_keys;
  delete [] input_vals;

  return 0;
}
