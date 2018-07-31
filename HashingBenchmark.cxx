
//We first check if VTKM_DEVICE_ADAPTER is defined, so that when TBB and CUDA
//includes this file we use the device adapter that they have set.
#ifndef VTKM_DEVICE_ADAPTER
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL
#endif

#include <cudpp_hash.h>
#include <cuda_util.h>
#include <cuda_profiler_api.h>
#include <mt19937ar.h>
#include <cuda_runtime_api.h>
#include "random_numbers.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/ArrayHandleDiscard.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayHandlePermutation.h>
#include <vtkm/cont/ArrayPortalToIterators.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/exec/ExecutionWholeArray.h>
#include <vtkm/worklet/HashFight.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>


typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;
typedef vtkm::cont::ArrayHandle<vtkm::Id> IdHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt64> LongHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::Float32> FloatHandleType;
typedef vtkm::cont::ArrayHandleIndex IndexHandleType;
typedef vtkm::cont::ArrayHandleConstant<vtkm::Id> ConstIdHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,2> > IdVecHandleType; 
typedef vtkm::cont::ArrayHandle<vtkm::Pair<vtkm::Id,vtkm::Id> > IdPairHandleType; 
typedef vtkm::cont::ArrayHandlePermutation<IdHandleType, IdHandleType>
                                             IdPermutationType;


#define __HASHING_BENCHMARK

// Macros for timing
//----------------------------------------------------------------------------
#if defined(__HASHING_BENCHMARK) && !defined(START_TIMER_BLOCK)
// start timer
# define START_TIMER_BLOCK(name) \
    vtkm::cont::Timer<DeviceAdapter> timer_##name;

// stop timer
# define END_TIMER_BLOCK(name) \
    std::cout << #name " : elapsed : " << timer_##name.GetElapsedTime() << "\n";
#endif
#if !defined(START_TIMER_BLOCK)
# define START_TIMER_BLOCK(name)
# define END_TIMER_BLOCK(name)
#endif


//#define DEBUG_PRINT

namespace debug 
{
#ifdef DEBUG_PRINT
//----------------------------------------------------------------------------
template<typename T, 
         typename S = VTKM_DEFAULT_STORAGE_TAG>
void 
HashingDebug(const vtkm::cont::ArrayHandle<T,S> &outputArray, 
		 const char* name)
{
  typedef T ValueType;
  typedef vtkm::cont::internal::Storage<T,S> StorageType;
  typedef typename StorageType::PortalConstType PortalConstType;
  PortalConstType readPortal = outputArray.GetPortalConstControl();
  vtkm::Id numElements = readPortal.GetNumberOfValues();
  std::cout << name << "= " << numElements << " [";
  for (vtkm::Id i = 0; i < numElements; i++)
    std::cout << readPortal.Get(i) << " ";
  std::cout << "]\n";
}
#else
template<typename T, typename S>
void 
HashingDebug(
    const vtkm::cont::ArrayHandle<T,S> &vtkmNotUsed(outputArray),
    const char* vtkmNotUsed(name))
{}
#endif
} // namespace debug


int main(int argc, char** argv)
{
  if (argc < 2)
    return -1;

  typedef vtkm::cont::DeviceAdapterTraits<DeviceAdapter> DeviceAdapterTraits;
  std::cout << "Running Hashing Benchmarks on device adapter: " 
            << DeviceAdapterTraits::GetName()
            << std::endl;


  //vtkm::Id image_width = (vtkm::Id)std::atoi(argv[3]);


//debug::HashingDebug(counts, "counts");


  std::cout << "========================CUDPP Cuckoo Hashing"
            << "==============================\n";

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
    printf("Using device %d:\n", 0);
    printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz; warps: %d; regs per block: %d; shared memory: %uB\n",
           prop.name, (unsigned int)prop.totalGlobalMem, (int)prop.major,
           (int)prop.minor, (int)prop.clockRate, (int)prop.warpSize, (int)prop.regsPerBlock, (unsigned int)prop.sharedMemPerBlock);
  }

  if (prop.major < 2)
  {
    fprintf(stderr, "ERROR: CUDPP hash tables are only supported on "
                "devices with compute\n  capability 2.0 or greater; "
                "exiting.\n");
    exit(1);
  }


  CUDPPHandle theCudpp;
  CUDPPResult result = cudppCreate(&theCudpp);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error initializing CUDPP Library.\n");
    exit(1);
  }

  //const unsigned kNumSpaceUsagesToTest = 5;
  const float kSpaceUsagesToTest[] = {1.05f, 1.15f, 1.25f, 1.5f, 2.0f};

  //unsigned int kInputSize = 100000000;
  
  unsigned int kInputSize = (unsigned int)std::atoi(argv[1]);

  CUDPPHashTableType htt = CUDPP_BASIC_HASH_TABLE;

  CUDPPHashTableConfig config;
  config.type = htt;
  config.kInputSize = kInputSize;
  //config.space_usage = kSpaceUsagesToTest[4];
  config.space_usage = (float)std::atof(argv[2]);  

  unsigned int* input_keys = new unsigned int[kInputSize];
  unsigned int* input_vals = new unsigned int[kInputSize];

  const unsigned int pool_size = kInputSize * 2;
  unsigned int* number_pool = new unsigned int[pool_size];
  GenerateUniqueRandomNumbers(number_pool, pool_size);

  // The unique numbers are pre-shuffled by the generator.
  // Take the first half as the input keys.
  memcpy(input_keys, number_pool, sizeof(unsigned int) * kInputSize);

  for (unsigned int i = 0; i < kInputSize; i++)
    input_vals[i] = genrand_int32();



#if 1

START_TIMER_BLOCK(CuckooHashingBuild)
  unsigned int* d_test_keys = NULL, *d_test_vals = NULL;
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

  CUDPPHandle hash_table_handle;
  result = cudppHashTable(theCudpp, &hash_table_handle, &config);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppHashTable call in"
                    "testHashTable (make sure your device is at"
                    "least compute version 2.0\n");
  }

  result = cudppHashInsert(hash_table_handle, d_test_keys, d_test_vals, kInputSize);
  cudaThreadSynchronize();
 

END_TIMER_BLOCK(CuckooHashingBuild)
 
  printf("Cuckoo Hash table build complete\n");
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppHashInsert call in"
                    "testHashTable\n");
  }

  //Begin querying phase for cuckoo hashing
  #if 0 
  const int failure_trials = 10;
  for (int failure = 0; failure <= failure_trials; ++failure)
  {
    // Generate a set of queries comprised of keys both
    // from and not from the input.
    float failure_rate = failure / (float) failure_trials;
    GenerateQueries(kInputSize, failure_rate, number_pool,
                                    query_keys);
    CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys,
                              sizeof(unsigned int) * kInputSize,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0,
                              sizeof(unsigned int) * kInputSize));

    /// --------------------------------------- Query the table.
    timer.reset();
    timer.start();
    unsigned int errors = 0;
    result = cudppHashRetrieve(hash_table_handle,
                               d_test_keys, d_test_vals,
                               kInputSize);
     
    
    cudaThreadSynchronize();
    timer.stop();
    if (result != CUDPP_SUCCESS)
      fprintf(stderr, "Error in cudppHashRetrieve call in"
                                "testHashTable\n");
    printf("\tHash table retrieve with %3u%% chance of "
           "failed queries: %f ms\n", failure * failure_trials,
                           timer.getTime());
  #endif
    

  /// -------------------------------------------- Free the table.
  result = cudppDestroyHashTable(theCudpp, hash_table_handle);
  if (result != CUDPP_SUCCESS)
  {
    fprintf(stderr, "Error in cudppDestroyHashTable call in"
                            "testHashTable\n");
  }
  CUDA_SAFE_CALL(cudaFree(d_test_keys));
  CUDA_SAFE_CALL(cudaFree(d_test_vals));
  delete [] number_pool;

  result = cudppDestroy(theCudpp);


  std::cout << "========================HashFight"
            << "==============================\n";

#if 0 
  delete [] input_keys;
  delete [] input_vals;
  input_keys = new unsigned int[20]; 
  input_vals = new unsigned int[20];
  kInputSize = 20;

  int i;
  unsigned int r;
  for (i = 0; i < kInputSize; i++)
  {
    r = genrand_int32();
    *(input_keys+i) = r;    
    *(input_vals+i) = r;
  }
#endif

#endif

  vtkm::cont::ArrayHandle<vtkm::UInt32> unsignedKeys =
    vtkm::cont::make_ArrayHandle(input_keys, kInputSize);
 
  vtkm::cont::ArrayHandle<vtkm::UInt32> unsignedVals =
    vtkm::cont::make_ArrayHandle(input_vals, kInputSize);

  debug::HashingDebug(unsignedKeys, "unsignedKeys");
  debug::HashingDebug(unsignedVals, "unsignedVals");
  
  /*
  vtkm::cont::ArrayHandleCast<vtkm::Id, vtkm::cont::ArrayHandle<unsigned int> > castArrayKeys(unsignedKeys);
  IdHandleType castedKeys;
  Algorithm::Copy(castArrayKeys, castedKeys);
 
  vtkm::cont::ArrayHandleCast<vtkm::Id, vtkm::cont::ArrayHandle<unsigned int> > castArrayVals(unsignedVals);
  IdHandleType castedVals;
  Algorithm::Copy(castArrayVals, castedVals);
 
  debug::HashingDebug(castedKeys, "castedKeys");
  debug::HashingDebug(castedVals, "castedVals");
  */

  LongHandleType hashTable;

  const vtkm::Float32 sizeFactor = (vtkm::Float32)std::atof(argv[2]);
  std::cout << "sizeFactor = " << sizeFactor << std::endl;

#if 1

  printf("size of unsigned long long: %lu\n", sizeof(unsigned long long));
  std::cerr << "Calling HashFight worklet..." << std::endl;
START_TIMER_BLOCK(HashFightBuild)

  //Run the HashFight worklet
  vtkm::worklet::HashFight().Run(
        unsignedKeys,
        unsignedVals,
        sizeFactor,
        hashTable,
        DeviceAdapter()
      );
 
END_TIMER_BLOCK(HashFightBuild)

  printf("HashFight hash table build complete\n");
 
  unsignedKeys.ReleaseResourcesExecution();
  unsignedVals.ReleaseResourcesExecution();

#endif

  cudaProfilerStop();
  //cudaDeviceReset();
  

  delete [] input_keys;
  delete [] input_vals;
  
}

#if (defined(VTKM_GCC) || defined(VTKM_CLANG))
# pragma GCC diagnostic pop
#endif
