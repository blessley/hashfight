#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include "tbb/tick_count.h"

#include <iostream>
#include <string>
#include <fstream>

#define __BUILDING_TBB_VERSION__ 

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"


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

#ifdef __BUILDING_TBB_VERSION__
  //Manually set the number of TBB threads invoked for this program
  char* numThreads = argv[7];
  if(numThreads == NULL)
  {
     printf("Define NUM_TBB_THREADS\n");
     exit(1);
  }
  int parallelism = std::atoi(numThreads);
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, parallelism);
#endif


  const std::string data_dir(argv[5]);
  double sortTime = 0, searchTime = 0;
  tbb::tick_count start, stop;

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
  start = tbb::tick_count::now();
  thrust::sort_by_key(keys_d.begin(), keys_d.end(), vals_d.begin());
  stop = tbb::tick_count::now();
  sortTime = (stop - start).seconds();
  //std::cout << "Sort: elapsed : " << sortTime/1000 << "\n";
  
  #if 1
  thrust::device_vector<unsigned int> query_keys_d(query_keys,
					           query_keys + kInputSize);
  
  thrust::device_vector<unsigned int> search_result(kInputSize);

  //std::cout << "Searching for pairs...\n";
  start = tbb::tick_count::now();
  thrust::binary_search(keys_d.begin(), keys_d.end(), 
                        query_keys_d.begin(), query_keys_d.end(),
                        search_result.begin());
  stop = tbb::tick_count::now();
  searchTime = (stop - start).seconds();
  
  delete [] query_keys;
  #endif

  //std::cout << "Search: elapsed : " << searchTime/1000 << "\n";
  std::cout << sortTime << "\n";
  std::cout << searchTime << "\n";

  delete [] input_keys;
  delete [] input_vals;

  return 0;
}
