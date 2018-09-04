
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
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

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
//#include <vtkm/worklet/HashFight.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>


typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;
typedef vtkm::cont::ArrayHandle<vtkm::Id> IdHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::Float32> FloatHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt32> UInt32HandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt64> UInt64HandleType;
typedef vtkm::cont::ArrayHandleIndex IndexHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt8> UInt8HandleType;
typedef vtkm::cont::ArrayHandleConstant<vtkm::UInt8> ConstBitHandleType;
typedef vtkm::cont::ArrayHandleConstant<vtkm::Id> ConstIdHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,2> > IdVecHandleType; 
typedef vtkm::cont::ArrayHandle<vtkm::Pair<vtkm::Id,vtkm::Id> > IdPairHandleType; 
typedef vtkm::cont::ArrayHandlePermutation<IdHandleType, IdHandleType>
                                             IdPermutationType;


constexpr vtkm::UInt32 emptyKey = 0xffffffffu;   //Signifies empty slots in the table.
constexpr vtkm::UInt32 keyNotFound = 0xffffffffu;   //Signifies that a query key was not found.

//Value indicating that a hash table slot has no valid item within it.
constexpr vtkm::UInt64 emptyEntry = vtkm::UInt64(emptyKey) << 32;


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
    std::cout << (vtkm::UInt64)readPortal.Get(i) << " ";
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


namespace hashfight
{
  
  template<typename DeviceAdapter,
           typename TableType>
  struct HashTable
  { 
    using Algorithm = vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;
 
    static constexpr vtkm::FloatDefault subTableSizeFactor = 1.575;
    //static constexpr vtkm::FloatDefault subTableSizeFactor = 1.4935;
    static constexpr vtkm::Id maxNumSubTables = 25;

    TableType entries;
    std::vector<vtkm::UInt32> sub_table_starts;
    vtkm::FloatDefault size_factor; 
    vtkm::Id size;
    vtkm::UInt32 num_keys;

    HashTable(const vtkm::UInt32 &k,
              const vtkm::FloatDefault &f) 
    : size_factor(f), num_keys(k)
    {
      size = (vtkm::Id)(subTableSizeFactor * size_factor * num_keys);

      //Initialize the subtable start indices to 0
      //sub_table_starts.assign(maxNumSubTables, (vtkm::Id)0);

      //Allocate the multi-level hash table
      Algorithm::Copy(vtkm::cont::make_ArrayHandleConstant(emptyEntry, size), entries);
    } 

  }; //struct HashTable

 
  class CollectHashStats : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableSize;
    vtkm::UInt32 SubTableStart;

    vtkm::UInt32 ChunkStart;
    vtkm::UInt32 ChunkEnd;

  public:
    typedef void ControlSignature(FieldIn<>, FieldOut<>);
    typedef void ExecutionSignature(WorkIndex, _1, _2);

    static constexpr vtkm::UInt32 FNV1A_OFFSET = 2166136261;
    static constexpr vtkm::UInt32 FNV1A_PRIME = 16777619;

    VTKM_CONT
    CollectHashStats(const unsigned int &size, 
                const unsigned int &start,
                const unsigned int &c_start,
                const unsigned int &c_end) 
      : SubTableSize(size), SubTableStart(start)
      {
        ChunkStart = c_start + SubTableStart;
        ChunkEnd = c_end + SubTableStart;
      }

    template<typename ActiveType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
	            ActiveType &isThisPass) const
    {
        vtkm::UInt32 h = (FNV1A_OFFSET ^ key) * FNV1A_PRIME;
        h = (h % SubTableSize) + SubTableStart;
        if (h >= ChunkStart && h < ChunkEnd)
          isThisPass = 1;
        else
          isThisPass = 0;
    }
  };


  VTKM_EXEC 
  inline vtkm::UInt32 FNV1aHash(const vtkm::UInt32 &x)
  { 
    static constexpr vtkm::UInt32 FNV1A_OFFSET = 2166136261;
    static constexpr vtkm::UInt32 FNV1A_PRIME = 16777619;

    #if 0
    const vtkm::UInt8 bytes[4] = {(x >> 24) & 0xFF,
			          (x >> 16) & 0xFF,
				  (x >> 8) & 0xFF,
				  x & 0xFF};
    //const char *data = (char*)x;
    vtkm::UInt32 h = FNV1A_OFFSET;
    h = (h ^ bytes[0]) * FNV1A_PRIME;
    h = (h ^ bytes[1]) * FNV1A_PRIME;
    h = (h ^ bytes[2]) * FNV1A_PRIME;
    h = (h ^ bytes[3]) * FNV1A_PRIME;
    #endif    

    return (FNV1A_OFFSET ^ x) * FNV1A_PRIME;
  }

  VTKM_EXEC 
  inline vtkm::UInt32 MurmurHash3(const vtkm::UInt32 &x)
  {
    vtkm::UInt32 h = x ^ (x >> 16);
    //h *= 0x85ebca6b;
    h *= 0x7feb352d;
    //h ^= h >> 13;
    h ^= h >> 15;
    //h *= 0xc2b2ae35;
    h *= 0x846ca68b;
    h ^= h >> 16;
    return h;
  }

  /* Hashes an array of keys into indices within the hash table.
   * The FNV1-a hash function is used to map an unsigned integer key
   * to another unsigned integer hash value, which is projected down
   * to the range of the hash table indices. The 32 bit key is then
   * prepended to its 32-bit value for a 64-bit combined entry, and then
   * written into the hash table at the hashed index. Since this is a
   * data-parallel operation, multiple threads may simulateously write
   * their entries to the same table index, leading to a collision. Since
   * no atomics are used, the last thread to write it's entry into the index
   * is the winner of the "hash fight", a non-deterministic scatter process. 
   */
  class ComputeHash : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableSize;
    vtkm::UInt32 SubTableStart;

    vtkm::UInt32 ChunkStart;
    vtkm::UInt32 ChunkEnd;

  public:
    typedef void ControlSignature(FieldIn<>, FieldIn<>, WholeArrayIn<>, WholeArrayInOut<>);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3, _4);

    VTKM_CONT
    ComputeHash(const unsigned int &size, 
                const unsigned int &start,
                const unsigned int &c_start,
                const unsigned int &c_end) 
      : SubTableSize(size), SubTableStart(start)
      {
        ChunkStart = c_start + SubTableStart;
        ChunkEnd = c_end + SubTableStart;
      }

    template<typename ActiveType,
	     typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
		    const vtkm::UInt32 &value,
	            ActiveType &isActive,
                    TableType &hashTable) const
    {
      if (isActive.Get(index))
      {
        //vtkm::UInt32 h = FNV1aHash(key);
        vtkm::UInt32 h = MurmurHash3(key);
        h = (h % SubTableSize) + SubTableStart;
        if (h >= ChunkStart && h < ChunkEnd)
          hashTable.Set(h, ((vtkm::UInt64)(key) << 32) + value);
      }
    }
  };

   
  //Worklet that detects whether a face is internal.  If the
  //face is internal, then a value should not be assigned to the
  //face in the output array handle of face vertices; only external
  //faces should have a vector not equal to <-1,-1,-1>
  class CheckForMatches : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableSize;
    vtkm::UInt32 SubTableStart;
 
    vtkm::UInt32 ChunkStart;
    vtkm::UInt32 ChunkEnd;
 
  public:
    typedef void ControlSignature(FieldIn<>, WholeArrayIn<>, WholeArrayInOut<>);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3);
 
    VTKM_CONT
    CheckForMatches(const unsigned int &size, 
                const unsigned int &start,
                const unsigned int &c_start,
                const unsigned int &c_end) 
      : SubTableSize(size), SubTableStart(start)
      {
        ChunkStart = c_start + SubTableStart;
        ChunkEnd = c_end + SubTableStart;
      }

    template<typename ActiveType,
             typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
		    TableType &hashTable,
                    ActiveType &isActive) const
    {
      if (isActive.Get(index))
      {    
        //vtkm::UInt32 h = FNV1aHash(key); 
        vtkm::UInt32 h = MurmurHash3(key);
        h = (h % SubTableSize) + SubTableStart;
        if (h >= ChunkStart && h < ChunkEnd)
        {
        vtkm::UInt64 winningEntry = hashTable.Get(h);
        vtkm::UInt32 winningKey = (vtkm::UInt32)(winningEntry >> 32);
        isActive.Set(index, (vtkm::UInt8)vtkm::Min(vtkm::UInt32(1), (winningKey ^ key)));
        }
      }
    }
  };


  class ProbeForKey : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableSize;
    vtkm::UInt32 SubTableStart;
 
    vtkm::UInt32 ChunkStart;
    vtkm::UInt32 ChunkEnd;
 
  public:
    typedef void ControlSignature(FieldIn<>, FieldInOut<>, WholeArrayIn<>, WholeArrayInOut<>);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3, _4);
  
    VTKM_CONT
    ProbeForKey(const unsigned int &size, 
                const unsigned int &start,
                const unsigned int &c_start,
                const unsigned int &c_end) 
      : SubTableSize(size), SubTableStart(start)
      {
        ChunkStart = c_start + SubTableStart;
        ChunkEnd = c_end + SubTableStart;
      }

    template<typename ActiveType,
             typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
	            vtkm::UInt32 &val,
  	            TableType &hashTable,
                    ActiveType &isActive) const
    {
      if (isActive.Get(index))
      {    
        //vtkm::UInt32 h = FNV1aHash(key);
        vtkm::UInt32 h = MurmurHash3(key);
        h = (h % SubTableSize) + SubTableStart; 
        if (h >= ChunkStart && h < ChunkEnd)
        {
          vtkm::UInt64 winningEntry = hashTable.Get(h);
          vtkm::UInt32 winningKey = (vtkm::UInt32)(winningEntry >> 32);
          vtkm::UInt32 isEmptyKey = winningKey ^ emptyKey; //If 0, then empty entry
          vtkm::UInt32 isMatch = winningKey ^ key; // If 0, then the key is found
          if (!isEmptyKey || !isMatch)
            isActive.Set(index, (vtkm::UInt8)(0));

          //vtkm::UInt8 isFound = (vtkm::UInt8)vtkm::Min(vtkm::UInt32(1), (winningKey ^ key));
          //isActive.Set(index, isFound);   
          if (!isMatch) 
            val = (vtkm::UInt32)winningEntry;
        }
      }
    }
  };

  /* Inserts a batch of key-value pairs into a hash table;
   * i.e., constructs a hash table to store the key-value
   * pairs for future queries/lookups.
   *
  */
  template<typename DeviceAdapter>
  vtkm::Float64 
  Insert(const UInt32HandleType &keys,
         const UInt32HandleType &vals,
         struct HashTable<DeviceAdapter,UInt64HandleType> &hash_table)
  {
    
    using Algorithm = vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

    vtkm::Float64 elapsedTime = 0;

    vtkm::UInt32 numActiveEntries = hash_table.num_keys;
    
    //The active keys that still haven't been placed in the table
    UInt8HandleType isActive;
    ConstBitHandleType allActive(vtkm::UInt8(1), numActiveEntries);
    Algorithm::Copy(allActive, isActive);

    vtkm::UInt32 subTableStart = 0;
    vtkm::UInt32 subTableSize = (vtkm::UInt32)(hash_table.size_factor * numActiveEntries);   
    vtkm::Id numLoops = 0;

    //Debug: for hash table usage statistics
    #if 1    
    vtkm::UInt64 totalSpaceUsed = subTableSize;
    #endif

    //Begin hash fighting to insert keys
    while (numActiveEntries > 0)
    {
 
      //std::cerr << "=============Loop " << numLoops << " =======================\n";
      //std::cerr << "numActiveEntries = " << numActiveEntries << "\n";
      //std::cerr << "subtableSize = " << subTableSize << "\n";
      //std::cerr << "subtableStart = " << subTableStart << "\n";
 
      //std::cout << "Starting ComputeHash\n";

      
      hash_table.sub_table_starts.push_back(subTableSize); 

      //vtkm::UInt32 minSize = 250000000;
      vtkm::UInt32 minSize = subTableSize;
      vtkm::Id numPasses = (vtkm::Id)vtkm::Ceil(subTableSize / (vtkm::Float32)minSize);
      vtkm::UInt32 chunkSize = (vtkm::UInt32)vtkm::Ceil(subTableSize / (vtkm::Float32)numPasses);;
      //std::cout << "numPasses = " << numPasses << "\n";
      //std::cout << "chunkSize = " << chunkSize << "\n"; 
      vtkm::UInt32 chunkStart = 0, chunkEnd = 0;
      if (subTableSize <= minSize)
      {
        numPasses = 1;
        //if (subTableSize <= 1000000)
          //break;
      }
      //vtkm::UInt32 totalPassKeys = 0;
      for (vtkm::Id pass = 0; pass < numPasses; pass++)
      {
        chunkStart = (vtkm::UInt32)(pass*chunkSize);
        chunkEnd = (vtkm::UInt32)vtkm::Min((vtkm::Id)subTableSize, (pass+1)*chunkSize);
        if (subTableSize <= minSize)
        {
          chunkStart = 0;
          chunkEnd = subTableSize;
        }
        //std::cout << "start = " << chunkStart << "\n";
        //std::cout << "end = " << chunkEnd << "\n";
      /*
      if (numLoops == 0) {
      UInt8HandleType isInThisPass;
      CollectHashStats statsWorklet(subTableSize, subTableStart, chunkStart, chunkEnd);
      vtkm::worklet::DispatcherMapField<CollectHashStats,DeviceAdapter> statsDispatcher(statsWorklet);
      statsDispatcher.Invoke(keys, isInThisPass); 
      vtkm::UInt32 numInPass = Algorithm::Reduce(vtkm::cont::make_ArrayHandleCast<vtkm::UInt32>(isInThisPass), 
						 (vtkm::UInt32)0);
      totalPassKeys += numInPass;
      std::cout << "pass " << pass << ": " << numInPass << " hashed keys\n"; 
      }
      */

      //Hash each key to an index in the hash table, with collisions likely occuring.
      //The last key to write to an index is the winner of the "hash fight".
      //No atomics are used to handle colliding writes - winner takes all approach.
      ComputeHash hashWorklet(subTableSize, subTableStart, chunkStart, chunkEnd);
      vtkm::worklet::DispatcherMapField<ComputeHash> hashDispatcher(hashWorklet);
      vtkm::cont::Timer<DeviceAdapter> scatterTimer;
      hashDispatcher.Invoke(keys, vals, isActive, hash_table.entries);
      elapsedTime += scatterTimer.GetElapsedTime();      

      //std::cout << "Starting CheckForMatches\n";

      //Check for the winners of the hash fight.
      //Successfully-hashed keys are marked as inactive and removed from future fights.

      }
      //std::cout << "Total hashed keys = " << totalPassKeys < "\n";
      debug::HashingDebug(keys, "keys");
      debug::HashingDebug(vals, "vals");
      debug::HashingDebug(isActive, "isActive");
      debug::HashingDebug(hash_table.entries, "hashTable");      
 
      //minSize = 1000000;
      
      //numPasses = 1;
      #if 1
      numPasses = (vtkm::Id)vtkm::Ceil(subTableSize / (vtkm::Float32)minSize);
      chunkSize = (vtkm::UInt32)vtkm::Ceil(subTableSize / (vtkm::Float32)numPasses);;
      if (subTableSize <= minSize)
        numPasses = 1;
      for (vtkm::Id pass = 0; pass < numPasses; pass++)
      {
        chunkStart = (vtkm::UInt32)(pass*chunkSize);
        chunkEnd = (vtkm::UInt32)vtkm::Min((vtkm::Id)subTableSize, (pass+1)*chunkSize);
        if (subTableSize <= minSize)
        {
          chunkStart = 0;
          chunkEnd = subTableSize;
        }
        //std::cout << "start = " << chunkStart << "\n";
        //std::cout << "end = " << chunkEnd << "\n";
     
        
      CheckForMatches matchesWorklet(subTableSize, subTableStart, chunkStart, chunkEnd);
      vtkm::worklet::DispatcherMapField<CheckForMatches> matchDispatcher(matchesWorklet);
      vtkm::cont::Timer<DeviceAdapter> gatherTimer;
      matchDispatcher.Invoke(keys,
			     hash_table.entries,
                             isActive);
      elapsedTime += gatherTimer.GetElapsedTime();
     }
     #endif
      debug::HashingDebug(isActive, "isActive");
     
      //Debug: for hash table collision statistics
      vtkm::cont::Timer<DeviceAdapter> reduceTimer;
      vtkm::UInt32 numLosers = Algorithm::Reduce(vtkm::cont::make_ArrayHandleCast<vtkm::UInt32>(isActive), 
						 (vtkm::UInt32)0);
      elapsedTime += reduceTimer.GetElapsedTime();
      //vtkm::UInt32 numWinners = (vtkm::UInt32) numActiveEntries - numLosers;
      //std::cout << "numLosers = " << numLosers << "\n";
      //std::cout << "numWinners = " << numWinners << "\n";
      //std::cout << "percent placed in table = " << numWinners / (1.0f*numActiveEntries) << "\n";
 
      numActiveEntries = numLosers;  
      subTableStart += subTableSize; 
      subTableSize = (vtkm::UInt32)(hash_table.size_factor * numActiveEntries);
      totalSpaceUsed += subTableSize;
      numLoops++;

      //std::cerr << "==================================================\n";

    }  //End of while loop
    
    //std::cout << "Total space used: " << totalSpaceUsed << "\n";
    //std::cout << "Total allocated space: " << hash_table.size << "\n";

    return elapsedTime;
  } //hashfight::Insert


  
  template<typename DeviceAdapter>
  vtkm::Float64 
  Query(const UInt32HandleType &query_keys,
        const struct HashTable<DeviceAdapter,UInt64HandleType> &ht,
        UInt32HandleType &query_values)
  {
 
    using Algorithm = vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

    vtkm::Float64 elapsedTime = 0;

    //vtkm::Id numActiveEntries = query_keys.GetNumberOfValues();
    vtkm::Id numSubTables = (vtkm::Id)ht.sub_table_starts.size();

    debug::HashingDebug(ht.entries, "hashTable");
    //std::cout << "numSubTables = " << numSubTables << "\n";
 
    UInt8HandleType isActive;
    ConstBitHandleType allActive(vtkm::UInt8(1), query_keys.GetNumberOfValues());
    Algorithm::Copy(allActive, isActive);

    vtkm::UInt32 subTableStart = 0;
    vtkm::UInt32 subTableSize = ht.sub_table_starts[0];   
    vtkm::UInt64 totalSpaceUsed = subTableSize;
    vtkm::Id numLoops = 0;

    //Begin querying
    while (numLoops < numSubTables)
    {     
      //std::cout << "=============Loop " << numLoops << " =======================\n";
      //std::cout << "numActiveEntries = " << numActiveEntries << "\n";
      //std::cout << "subTableStart = " << subTableStart << "\n";
      //std::cout << "subTableSize = " << subTableSize << "\n";


      //std::cout << "Starting ProbeForKey\n";

      //const vtkm::UInt32 minSize = 250000000;
      const vtkm::UInt32 minSize = subTableSize;
      vtkm::Id numPasses = (vtkm::Id)vtkm::Ceil(subTableSize / (vtkm::Float32)minSize);
      const vtkm::UInt32 chunkSize = (vtkm::UInt32)vtkm::Ceil(subTableSize / (vtkm::Float32)numPasses);
      //std::cout << "numPasses = " << numPasses << "\n";
      //std::cout << "chunkSize = " << chunkSize << "\n"; 
      vtkm::UInt32 chunkStart = 0, chunkEnd = 0;
      //const vtkm::UInt32 minSize = subTableSize;
      //vtkm::UInt32 numActiveEntries = (vtkm::UInt32)(subTableSize / ht.size_factor);
      //std::cout << "numActiveEntries = " << numActiveEntries << "\n";
      if (subTableSize <= minSize)
      {
        numPasses = 1;
        //if (subTableSize <= 1000000)
          //break;
      }
      //vtkm::UInt32 totalPassKeys = 0;
      for (vtkm::Id pass = 0; pass < numPasses; pass++)
      {
        chunkStart = (vtkm::UInt32)(pass*chunkSize);
        chunkEnd = (vtkm::UInt32)vtkm::Min((vtkm::Id)subTableSize, (pass+1)*chunkSize);
        if (subTableSize <= minSize)
        {
          chunkStart = 0;
          chunkEnd = subTableSize;
        }
        //std::cout << "start = " << chunkStart << "\n";
        //std::cout << "end = " << chunkEnd << "\n";

      ProbeForKey queryWorklet(subTableSize, subTableStart, chunkStart, chunkEnd);
      vtkm::worklet::DispatcherMapField<ProbeForKey> queryDispatcher(queryWorklet);
      vtkm::cont::Timer<DeviceAdapter> queryTimer;
      queryDispatcher.Invoke(query_keys,
			     query_values,
			     ht.entries,
                             isActive);
      elapsedTime += queryTimer.GetElapsedTime();
      }

      debug::HashingDebug(isActive, "isActive");
      debug::HashingDebug(query_keys, "query_keys");
      debug::HashingDebug(query_values, "query_values");

      /*
      vtkm::UInt32 numLosers = Algorithm::Reduce(vtkm::cont::make_ArrayHandleCast<vtkm::UInt32>(isActive), 
						 (vtkm::UInt32)0);
      vtkm::UInt32 numWinners = (vtkm::UInt32) numActiveEntries - numLosers;
      std::cout << "numLosers = " << numLosers << "\n";
      std::cout << "numWinners = " << numWinners << "\n";
      std::cout << "percent found in table = " << numWinners / (1.0f*numActiveEntries) << "\n";
 
      numActiveEntries = numLosers;  
      */      

      subTableStart += subTableSize;
      numLoops++;
      subTableSize = ht.sub_table_starts[numLoops];
      totalSpaceUsed += subTableSize;

      //std::cout << "==================================================\n";
    }

    //std::cout << "Total space used: " << totalSpaceUsed << "\n";
    //std::cout << "Total allocated space: " << ht.size << "\n";

    return elapsedTime;
  }
  
} //namespace hashfight


int CheckResults_basic(const unsigned kInputSize,
                       const std::unordered_map<unsigned, unsigned> &pairs,
                       const unsigned *query_keys,
                       const unsigned *query_vals)
{
  int errors = 0;
  for (unsigned i = 0; i < kInputSize; ++i)
  {
    unsigned actual_value = keyNotFound;
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
  const unsigned int maxInputSize = 1300000000;
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




  std::cout << "========================CUDPP Cuckoo Hashing"
            << "==============================\n";
  
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

  std::cout << argv[1] << " pairs; " << argv[2] << " load factor; " 
            << std::atof(argv[3])/std::atoi(argv[4]) << " query failure rate\n"; 
  #if 1
  std::cout << "Loading binary of input keys...\n";
  load_binary(input_keys, kInputSize, data_dir + "/inputKeys-"+std::string(argv[1])+"-"+std::string(argv[5])); 

  std::cout << "Loading binary of input values...\n";
  load_binary(input_vals, kInputSize, data_dir + "/inputVals-"+std::string(argv[1])+"-"+std::string(argv[5])); 
  #endif  

    
  #if 1
  //Generate a set of queries comprised of keys both
  //from and not from the input.
  std::cout << "Loading binary of query keys...\n";
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
 
START_TIMER_BLOCK(CuckooInsert) 
  result = cudppHashInsert(hash_table_handle, d_test_keys, d_test_vals, kInputSize);
  cudaThreadSynchronize();  
END_TIMER_BLOCK(CuckooInsert) 

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


START_TIMER_BLOCK(CuckooQuery) 
  result = cudppHashRetrieve(hash_table_handle,
                               d_test_keys, d_test_vals,
                               kInputSize); 
  cudaThreadSynchronize();
END_TIMER_BLOCK(CuckooQuery)
 
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

#if 1
  using Algorithm = vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter>;

  std::cout << "========================VTK-m HashFight Hashing"
            << "==============================\n";
  //std::cout << "Running on device adapter: " << DeviceAdapterTraits::GetName() << "\n";

  vtkm::cont::ArrayHandle<vtkm::UInt32> insertKeys =
    vtkm::cont::make_ArrayHandle(input_keys, kInputSize);
 
  vtkm::cont::ArrayHandle<vtkm::UInt32> insertVals =
    vtkm::cont::make_ArrayHandle(input_vals, kInputSize);

  debug::HashingDebug(insertKeys, "insertKeys");
  debug::HashingDebug(insertVals, "insertVals"); 

  
  insertKeys.PrepareForInput(DeviceAdapter());
  insertVals.PrepareForInput(DeviceAdapter());
  
  //Configure and initialize the hash table
  hashfight::HashTable<DeviceAdapter,UInt64HandleType> ht((vtkm::Id)kInputSize,
                                                        (vtkm::FloatDefault)std::atof(argv[2]));
  //Insert the keys into the hash table 
  //std::cout << "(HashFight) Inserting into hash table...\n";
  vtkm::Float64 elapsedTime;
  elapsedTime = hashfight::Insert<DeviceAdapter>(insertKeys,
                                   insertVals,
                                   ht);

  std::cout << "HashFightInsert : elapsed : " << elapsedTime << "\n";
  debug::HashingDebug(ht.entries, "hashTable");
 
  insertKeys.ReleaseResourcesExecution();
  insertVals.ReleaseResourcesExecution();

  //Begin query phase
  vtkm::cont::ArrayHandle<vtkm::UInt32> queryKeys =
    vtkm::cont::make_ArrayHandle(query_keys, kInputSize);

  queryKeys.PrepareForInput(DeviceAdapter());

  vtkm::cont::ArrayHandle<vtkm::UInt32> queryVals;
  Algorithm::Copy(vtkm::cont::make_ArrayHandleConstant(keyNotFound, kInputSize), queryVals);

  debug::HashingDebug(queryKeys, "queryKeys");
  debug::HashingDebug(queryVals, "queryVals");


  //printf("(HashFight) Querying with %.3f chance of failed queries...\n", failure_rate);
  //Query the hash table
  elapsedTime = hashfight::Query<DeviceAdapter>(queryKeys,
                                  ht,
                                  queryVals);

  std::cout << "HashFightQuery : elapsed : " << elapsedTime << "\n";
  debug::HashingDebug(queryKeys, "queryKeys");
  debug::HashingDebug(queryVals, "queryVals");
 

  #if 0
  errors = CheckResults_basic(kInputSize,
                                  pairs_basic,
                                  query_keys,
                                  vtkm::cont::ArrayPortalToIteratorBegin(queryVals.GetPortalConstControl()));
  if (errors > 0)
    printf("%d errors found\n", errors);
  else
    printf("No errors found, test passes\n");

  #endif
 
  queryVals.ReleaseResources();
  queryKeys.ReleaseResources();
  insertKeys.ReleaseResources();
  insertVals.ReleaseResources();

#endif
  //cudaProfilerStop();
  
  delete [] number_pool;
  //delete [] input_keys;
  //delete [] input_vals;
  //delete [] query_keys; 
  //delete [] query_vals;

}

#if (defined(VTKM_GCC) || defined(VTKM_CLANG))
# pragma GCC diagnostic pop
#endif
