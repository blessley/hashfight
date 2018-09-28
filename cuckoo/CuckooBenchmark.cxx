
#include <cudpp_hash.h>
#include <cuda_util.h>
//#include <cuda_profiler_api.h>

#include <cuda_runtime.h>
#include <mt19937ar.h>
//#include <cuda_runtime_api.h>
#include "random_numbers.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>


template <typename T>
void dump_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write((char*) data, sizeof(T)*length);
    ofile.close();
}

template <typename T>
void load_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ifstream ifile(filename, std::ios::binary);
    ifile.read((char*) data, sizeof(T)*length);
    ifile.close();
}

int CheckResults_basic(const unsigned kInputSize,
                       const std::unordered_map<unsigned, unsigned> &pairs,
                       const unsigned *query_keys,
                       const unsigned *query_vals)
{
  int errors = 0;
  for (unsigned i = 0; i < kInputSize; ++i)
  {
    unsigned actual_value = 0xffffffffu;
    std::unordered_map<unsigned, unsigned>::const_iterator it =
            pairs.find(query_keys[i]);
    if (it != pairs.end())
      actual_value = it->second;
    if (actual_value != query_vals[i])
    {
      errors++;
      printf("\t\t\tError for key %10u: Actual value is "
                   "%10u, but hash returned %10u.\n",
                   query_keys[i], actual_value, query_vals[i]);
    }
  }
  return errors;
}


int main(int argc, char** argv)
{
  if (argc < 2)
    return -1;

  
  std::string data_dir(argv[6]);

#if 0
  unsigned int* input_keys = NULL;
  unsigned int* input_vals = NULL;
  unsigned *query_keys = NULL;
  unsigned int pool_size = 0;
  unsigned int* number_pool = NULL;
  
  const int overall_trials = 1; 
  const int failure_trials = 10; 
  float failure_rate = 0.0f;
  const unsigned int maxInputSize = 1450000000;
  const unsigned int minInputSize = 550000000;
  const unsigned int inputStepSize = 50000000;
  //const int numSpaceUsagesToTest = 9;
  //const float kSpaceUsagesToTest[9] = {1.03f, 1.05f, 1.10f, 1.15f, 1.25f, 1.5f, 1.75f, 1.9f, 2.0f};
  for (int trialId = 0; trialId < overall_trials; trialId++)
  {
    std::cout << "---------------Trial # " << trialId << "-----------------\n";
    for (unsigned int kInputSize = maxInputSize; kInputSize >= minInputSize; kInputSize -= inputStepSize)
    {
      std::cout << "Input Size = " << kInputSize << "\n";
      pool_size = kInputSize * 2;
      input_keys = new unsigned int[kInputSize];
      input_vals = new unsigned int[kInputSize];
      number_pool = new unsigned int[pool_size];

      std::cout << "Generating random input keys\n";
      //Randomly-generate the input keys and values
      GenerateUniqueRandomNumbers(number_pool, pool_size);
  
      //The unique numbers are pre-shuffled by the generator.
      //Take the first half as the input keys.
      memcpy(input_keys, number_pool, sizeof(unsigned int) * kInputSize);

      std::cout << "Generating random input vals\n";
      for (unsigned int i = 0; i < kInputSize; i++)
        input_vals[i] = (unsigned int) genrand_int32();

      std::cout << "Dumping binary of input keys\n";
      dump_binary(input_keys, kInputSize, data_dir + "inputKeys-" + std::to_string(kInputSize) + 
     					"-" + std::to_string(trialId));

      std::cout << "Dumping binary of input vals\n";
      dump_binary(input_vals, kInputSize, data_dir + "inputVals-" + std::to_string(kInputSize) +   
 				      "-" + std::to_string(trialId));
     
      //Randomly-generate the query keys
      for (int failure = 0; failure < failure_trials; failure++)
      {
         failure_rate = failure / (float)failure_trials;
         std::cout << "Failure Rate = " << failure_rate << "\n";
         query_keys = new unsigned int[kInputSize];
         std::cout << "Generating random query keys\n";
         GenerateQueries(kInputSize, failure_rate, number_pool, query_keys); 
         std::cout << "Dumping binary of query keys\n";
         dump_binary(query_keys, kInputSize, data_dir + "queryKeys-" + std::to_string(kInputSize) +
 				            "-" + std::to_string(failure) + "-" +
				            std::to_string(failure_trials) + "-" +
				            std::to_string(trialId));
         delete [] query_keys;
       } 

       delete [] number_pool;
       delete [] input_keys;
       delete [] input_vals;
     } 
  }

#endif


#if 1

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "Error (main): no devices supporting CUDA.\n");
    exit(1);
  }
 
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) == 0)
  {
    //printf("Using device %d:\n", 0);
    //printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz; warps: %d; regs per block: %d; shared memory: %uB\n",
      //     prop.name, (unsigned int)prop.totalGlobalMem, (int)prop.major,
        //   (int)prop.minor, (int)prop.clockRate, (int)prop.warpSize, (int)prop.regsPerBlock, (unsigned int)prop.sharedMemPerBlock);
  }

  if (prop.major < 2)
  {
    fprintf(stderr, "ERROR: CUDPP hash tables are only supported on "
                "devices with compute\n  capability 2.0 or greater; "
                "exiting.\n");
    exit(1);
  }

  float cudppInsertTime = 0.0, cudppQueryTime = 0.0;

  //std::cout << "========================CUDPP Cuckoo Hashing"
    //       << "==============================\n";
  
  unsigned int kInputSize = (unsigned int)std::atoi(argv[1]);
  unsigned int* input_keys = new unsigned int[kInputSize];
  unsigned int* input_vals = new unsigned int[kInputSize];
  unsigned *query_keys = new unsigned[kInputSize];
  unsigned int pool_size = kInputSize * 2;
  unsigned int* number_pool = new unsigned int[pool_size];

  CUDPPHandle theCudpp;
  CUDPPResult result = cudppCreate(&theCudpp);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error initializing CUDPP Library.\n");
    exit(1);
  } 

  CUDPPHashTableType htt = CUDPP_BASIC_HASH_TABLE;

  CUDPPHashTableConfig config;
  config.type = htt;
  config.kInputSize = kInputSize;
  config.space_usage = (float)std::atof(argv[2]);  

  //std::cout << argv[1] << " pairs; " << argv[2] << " load factor; " 
    //        << std::atof(argv[3])/std::atoi(argv[4]) << " query failure rate\n"; 
  #if 1
  //std::cout << "Loading binary of input keys...\n";
  load_binary(input_keys, kInputSize, data_dir + "/inputKeys-"+std::string(argv[1])+"-"+std::string(argv[5])); 

  //std::cout << "Loading binary of input values...\n";
  load_binary(input_vals, kInputSize, data_dir + "/inputVals-"+std::string(argv[1])+"-"+std::string(argv[5])); 
  #endif  

    
  #if 1
  //Generate a set of queries comprised of keys both
  //from and not from the input.
  //std::cout << "Loading binary of query keys...\n";
  load_binary(query_keys, kInputSize, data_dir + "/queryKeys-"+std::string(argv[1])+"-"+std::string(argv[3])+"-"+std::string(argv[4])+"-"+std::string(argv[5]));
  #endif

  //std::cout << "Saving key-val pairs...\n";
  //Save the original input for checking the results.
 
  #if 0
  std::unordered_map<unsigned, unsigned> pairs_basic;
  for (unsigned i = 0; i < kInputSize; ++i)
    pairs_basic[input_keys[i]] = input_vals[i];
  #endif
  
  unsigned int* d_test_keys = NULL, *d_test_vals = NULL;

  //use nvprof flag: --profile-from-start off
  //cudaProfilerStart();

  //Begin insertion phase
  
  //std::cout << "cudaMallocs and cudaMemcpy calls...\n";
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys,
                            sizeof(unsigned int) * kInputSize));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals,
                            sizeof(unsigned int) * kInputSize));
 
  CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys,
                            sizeof(unsigned int) * kInputSize,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMemcpy(d_test_vals, input_vals,
                            sizeof(unsigned int) * kInputSize,
                            cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;

  //std::cout << "Creating the cudpp hash table...\n";
  CUDPPHandle hash_table_handle;
  result = cudppHashTable(theCudpp, &hash_table_handle, &config);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppHashTable call in"
                    "testHashTable (make sure your device is at"
                    "least compute version 2.0\n");
  }

  //std::cout << "(CuckooHash) Inserting into hash table...\n";
 
 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  result = cudppHashInsert(hash_table_handle, d_test_keys, d_test_vals, kInputSize);
  cudaDeviceSynchronize();   
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cudppInsertTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppHashInsert call in"
                    " testHashTable\n");
  }


  //Begin querying phase
 
  CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys,
                              sizeof(unsigned int) * kInputSize,
                              cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0,
                              sizeof(unsigned int) * kInputSize));

  //printf("(CuckooHash) Querying with %.3f chance of "
    //       "failed queries...\n", failure_rate);


//START_TIMER_BLOCK(CuckooQuery) 
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  result = cudppHashRetrieve(hash_table_handle,
                               d_test_keys, d_test_vals,
                               kInputSize); 
  cudaDeviceSynchronize();
//END_TIMER_BLOCK(CuckooQuery)

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cudppQueryTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (result != CUDPP_SUCCESS)
    fprintf(stderr, "Error in cudppHashRetrieve call in"
                                "testHashTable\n");


  #if 0  
  unsigned *query_vals = new unsigned[kInputSize];

  CUDA_SAFE_CALL(cudaMemcpy(query_vals, d_test_vals,
                              sizeof(unsigned) * kInputSize,
                              cudaMemcpyDeviceToHost));
  //Check the query results.    
  unsigned int errors = CheckResults_basic(kInputSize,
                                 pairs_basic,
                                 query_keys,
                                 query_vals);
  if (errors > 0)
    printf("%d errors found\n", errors);
  else
    printf("No errors found, test passes\n");
  #endif
   
  //Free the hash table and data arrays from the device
  result = cudppDestroyHashTable(theCudpp, hash_table_handle);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppDestroyHashTable call in"
                            "testHashTable\n");
  }
  CUDA_SAFE_CALL(cudaFree(d_test_keys));
  CUDA_SAFE_CALL(cudaFree(d_test_vals));
  result = cudppDestroy(theCudpp);
  if (result != CUDPP_SUCCESS)
    printf("Error shutting down CUDPP Library.\n");

#endif

  //cudaProfilerStop();
  std::cout << cudppInsertTime/1000 << "\n";
  std::cout << cudppQueryTime/1000 << "\n";
  

  delete [] number_pool;
  delete [] input_keys;
  delete [] input_vals;
  delete [] query_keys; 
  //delete [] query_vals;

}

