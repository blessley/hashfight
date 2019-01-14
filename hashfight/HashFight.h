//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================
#ifndef vtk_m_worklet_HashFight_h
#define vtk_m_worklet_HashFight_h


//#include <cuda_runtime_api.h>

#include <vtkm/Math.h>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/ArrayHandleConcatenate.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandleGroupVec.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayHandlePermutation.h>
#include <vtkm/cont/DeviceAdapterAlgorithm.h>
#include <vtkm/cont/Field.h>
#include <vtkm/cont/Timer.h>

#include <vtkm/exec/ExecutionWholeArray.h>

#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/DispatcherMapTopology.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/WorkletMapTopology.h>


typedef vtkm::cont::ArrayHandle<vtkm::Id> IdHandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt64> UInt64HandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt32> UInt32HandleType;
typedef vtkm::cont::ArrayHandle<vtkm::UInt8> UInt8HandleType;
typedef vtkm::cont::ArrayHandleIndex IndexHandleType;
typedef vtkm::cont::ArrayHandleConstant<vtkm::UInt8> ConstBitHandleType;
typedef vtkm::cont::ArrayHandleConcatenate<IdHandleType, IdHandleType> ConcatHandleType;
typedef vtkm::cont::ArrayHandleConcatenate<ConstBitHandleType, ConstBitHandleType> ConcatConstHandleType;
typedef vtkm::cont::ArrayHandlePermutation<IdHandleType, IdHandleType> IdPermuteHandleType;
typedef vtkm::cont::ArrayHandlePermutation<IdPermuteHandleType, IdHandleType> NestedIdPermuteHandleType;


const vtkm::UInt32 emptyKey = 0xffffffffu;   //Signifies empty slots in the table.
const vtkm::UInt32 keyNotFound = 0xffffffffu;   //Signifies that a query key was not found.

//Value indicating that a hash table slot has no valid item within it.
const vtkm::UInt64 emptyEntry = vtkm::UInt64(emptyKey) << 32;

//Value returned when a query fails.
const vtkm::UInt64 entryNotFound = (vtkm::UInt64(emptyKey) << 32) + keyNotFound;

//Returns the value of an Entry.
//vtkm::UInt32 value = (vtkm::UInt32)(entry & 0xffffffff);


//#define DEBUG_PRINT

namespace vtkm
{
namespace worklet
{

namespace debug 
{
#ifdef DEBUG_PRINT
//----------------------------------------------------------------------------
template<typename T, 
         typename S = VTKM_DEFAULT_STORAGE_TAG>
void 
HashFightDebug(const vtkm::cont::ArrayHandle<T,S> &outputArray, 
		 const char* name)
{
  typedef T ValueType;
  typedef vtkm::cont::internal::Storage<T,S> StorageType;
  typedef typename StorageType::PortalConstType PortalConstType;
  PortalConstType readPortal = outputArray.GetPortalConstControl();
 /*
  vtkm::cont::ArrayPortalToIterators<PortalConstType> iterators(readPortal);
  std::vector<ValueType> result(readPortal.GetNumberOfValues());
  std::copy(iterators.GetBegin(), iterators.GetEnd(), result.begin());
  std::copy(result.begin(), result.end(), std::ostream_iterator<ValueType>(std::cout, " "));
  */ 
  vtkm::Id numElements = readPortal.GetNumberOfValues();
  std::cout << name << "= " << numElements << " [";
  for (vtkm::Id i = 0; i < numElements; i++)
    std::cout << (vtkm::UInt64)readPortal.Get(i) << " ";
  std::cout << "]\n";
}
#else
template<typename T, typename S>
void 
HashFightDebug(
    const vtkm::cont::ArrayHandle<T,S> &vtkmNotUsed(outputArray),
    const char* vtkmNotUsed(name))
{}
#endif
} // namespace debug


struct HashFight
{
  //Unary predicate operator
  //Returns True if the argument is equal to the constructor
  //integer argument; False otherwise.
  struct IsIntValue
  {
  private:
    int Value;

  public:
    VTKM_CONT
    IsIntValue(const int &v) : Value(v) { };

    template<typename T>
    VTKM_EXEC
    bool operator()(const T &x) const
    {
      return x == T(Value);
    }
  };

  
  class CopyToSubTable : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableStart;

  public:
    typedef void ControlSignature(FieldIn, FieldIn, WholeArrayInOut);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3);

    VTKM_CONT
    CopyToSubTable(const unsigned int &start) : SubTableStart(start) { };
 
    template<typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
		    const vtkm::UInt32 &value,
                    TableType &hashTable) const
    {
      hashTable.Set(SubTableStart + index, ((vtkm::UInt64)(key) << 32) + value);
    }
  };

 
  //Worklet that calculates a hash key for the sorted points
  //of a face.  The key is an index into a hash table.
  class ComputeHash : public vtkm::worklet::WorkletMapField
  {
  private:
    vtkm::UInt32 SubTableSize;
    vtkm::UInt32 SubTableStart;
    vtkm::UInt32 ShiftBits;

  public:
    typedef void ControlSignature(FieldIn, FieldIn, WholeArrayIn, WholeArrayInOut);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3, _4);

    static constexpr vtkm::UInt32 FNV1A_OFFSET = 2166136261;
    static constexpr vtkm::UInt32 FNV1A_PRIME = 16777619;
    static constexpr vtkm::UInt32 MULT_CONST = 2654435769;

    VTKM_CONT
    ComputeHash(const unsigned int &size, 
                const unsigned int &start) : SubTableSize(size), SubTableStart(start) 
    { 
      //ShiftBits = (vtkm::UInt32)32 - (vtkm::UInt32)vtkm::Log2(SubTableSize);
    };

    #if 0
    template<typename EntryType,
             typename HashType,
	     typename TableType>
    VTKM_EXEC
    void operator()(const EntryType &entry,
		    HashType &hash,
                    TableType &hashTable) const
    {
      const vtkm::IdComponent numComps = EntryType::NUM_COMPONENTS;
      vtkm::UInt32 h, key;
      for (vtkm::IdComponent i = 0; i < numComps; ++i)
      {
        key = (vtkm::UInt32)(entry[i] >> 32);
        h = (FNV1A_OFFSET * FNV1A_PRIME) ^ key;
        hash[i] = (vtkm::Id)((h % SubTableSize) + SubTableStart); 
        hashTable.Set(hash[i], entry[i]);
      }
    }
    #endif

    #if 0 
    template<typename KeyType, 
             typename HashType,
             typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id threadIdx,
                    KeyType &entries,
		    HashType &hashes,
                    TableType &hashTable) const
    {
      vtkm::UInt32 h, key; 
      vtkm::Id offset;
      vtkm::UInt64 entry = entries.Get(threadIdx/32);
      key = (vtkm::UInt32)(entry >> 32);
      h = (FNV1A_OFFSET * FNV1A_PRIME) ^ key;
      h = (h % SubTableSize) + SubTableStart;
      hashes.Set(threadIdx/32, (vtkm::Id)h);
      offset = threadIdx % 32;
      if (SubTableSize-32-h < 0)
        offset *= -1;
      hashTable.Set((vtkm::Id)h+offset, entry);
    }
    #endif
 
    #if 1 
    template<typename ActiveType,
	     typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
		    const vtkm::UInt32 &value,
	            ActiveType &isActive,
   	            //vtkm::UInt32 &hash,
                    TableType &hashTable) const
    {
      if (isActive.Get(index))
      {
        /*
        vtkm::UInt32 h = FNV1A_OFFSET;
        for (vtkm::Id i = 0; i < 4; i++)
        {
          h = h ^ ( (key >> (i*8)) & 0xff);
          h *= FNV1A_PRIME;          
        }
        */
        vtkm::UInt32 h = (FNV1A_OFFSET ^ key) * FNV1A_PRIME;
        //hash = ((vtkm::UInt64)key * (vtkm::UInt64)SubTableSize) >> 32; 
        //hash += SubTableStart;
        //h ^= (h >> ShiftBits);
        //hash = ((MULT_CONST * h) >> ShiftBits) + SubTableStart;
        h = (h % SubTableSize) + SubTableStart;
        hashTable.Set(h, ((vtkm::UInt64)(key) << 32) + value);
      }
    }
    #endif
  };

  class GetEntry : public vtkm::worklet::WorkletMapField
  {
  public:
    typedef void ControlSignature(FieldIn, FieldIn, FieldOut);
    typedef void ExecutionSignature(_1, _2, _3);

    VTKM_CONT
    GetEntry() { }

    #if 0
    template<typename KeyType,
             typename ValType,
             typename EntryType>
    VTKM_EXEC
    void operator()(const KeyType &key,
		    const ValType &value,
                    EntryType &entry) const
    {    
      const vtkm::IdComponent numComps = KeyType::NUM_COMPONENTS;
      for (vtkm::IdComponent i = 0; i < numComps; ++i)
      {
        //Makes an 64-bit entry out of the key-value pair
        entry[i] = ((vtkm::UInt64)(key[i]) << 32) + value[i];
      }
    }
    #endif

    #if 1
    VTKM_EXEC
    void operator()(const vtkm::UInt32 &key,
		    const vtkm::UInt32 &value,
                    vtkm::UInt64 &entry) const
    {    
      //Makes an 64-bit entry out of the key-value pair
      entry = ((vtkm::UInt64)(key) << 32) + value;
    }
    #endif

  };


  //Worklet that writes the face index, i, at
  //location hashes[i] of the hash table
  class Scatter : public vtkm::worklet::WorkletMapField
  {
  public:
    typedef void ControlSignature(FieldIn, FieldIn, WholeArrayInOut);
    typedef void ExecutionSignature(_1, _2, _3);

    VTKM_CONT
    Scatter() { }

    template<typename T>
    VTKM_EXEC
    void operator()(const vtkm::Id &hash,
                    const vtkm::UInt64 &entry,
                    T &hashTable) const
    {    
      hashTable.Set(hash, entry);
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
 
  public:
    typedef void ControlSignature(FieldIn, WholeArrayIn, WholeArrayInOut);
    typedef void ExecutionSignature(WorkIndex, _1, _2, _3);
 
    static constexpr vtkm::UInt32 FNV1A_OFFSET = 2166136261;
    static constexpr vtkm::UInt32 FNV1A_PRIME = 16777619;

    VTKM_CONT
    CheckForMatches(const unsigned int &size,
	            const unsigned int &start): SubTableSize(size), SubTableStart(start) { }

    template<typename ActiveType,
             typename TableType>
    VTKM_EXEC
    void operator()(const vtkm::Id &index,
                    const vtkm::UInt32 &key,
		    //const vtkm::UInt32 &hash,
		    TableType &hashTable,
                    //const vtkm::Id &winningKeyId,
                    //const vtkm::Id &keyId,
                    ActiveType &isActive) const
    {
      if (isActive.Get(index))
      {    
        vtkm::UInt32 h = (FNV1A_OFFSET ^ key) * FNV1A_PRIME;
        h = (h % SubTableSize) + SubTableStart;
        vtkm::UInt64 winningEntry = hashTable.Get(h);
        //Returns the key of an entry.
        vtkm::UInt32 winningKey = (vtkm::UInt32)(winningEntry >> 32);
        isActive.Set(index, (vtkm::UInt8)vtkm::Min(vtkm::UInt32(1), (winningKey ^ key)));

      //Key equality: this key is the winning key of the hash fight or its duplicate
      //if (winningKey == thisKey)
      //{
        //isActive = vtkm::UInt8(0);
        /*
        if (winningKeyId == keyId)
        {
          isInactive = vtkm::UInt8(1);
        }
        else
        {
          isInactive = vtkm::UInt8(1);
          isDuplicate.Set(keyId, vtkm::UInt8(1));
          isDuplicate.Set(winningKeyId, vtkm::UInt8(1));
        }
        */
      //}
      }
    }
  };


  
#if 1 
  template<typename DeviceAdapter>
  void 
  Query(const UInt32HandleType &query_keys,
        const UInt64HandleType &hash_table,
        const vtkm::Float32 &table_size_factor,
        const vtkm::Id &num_queries,
        const vtkm::Id &table_size,
        const std::vector<vtkm::Id> &sub_table_starts,
        UInt32HandleType &query_values)
  {
 
    typedef vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter> Algorithm;

    vtkm::Id numActiveEntries = num_queries;
    
    std::cout << "numEntries: " << numActiveEntries << "\n";
    std::cout << "sizeFactor: " << table_size_factor << "\n";
 
    UInt8HandleType isActive;
    ConstBitHandleType allActive(vtkm::UInt8(1), numActiveEntries);
    Algorithm::Copy(allActive, isActive);

    UInt32HandleType tempKeys, tempVals;
    vtkm::Id subTableStart = 0;
    vtkm::Id subTableSize = sub_table_starts[1];   
    vtkm::UInt64 totalSpaceUsed = subTableSize;
    vtkm::Id numLoops = 0;

    //Begin querying
    while (numActiveEntries > 0)
    {    
      std::cout << "Starting CheckForMatches\n";
      CheckForMatches matchesWorklet(subTableSize, subTableStart);
      vtkm::worklet::DispatcherMapField<CheckForMatches, DeviceAdapter> matchDispatcher(matchesWorklet);
      matchDispatcher.Invoke(query_keys,
			     hash_table,
                             isActive);

      debug::HashFightDebug(isActive, "isActive");

      vtkm::UInt32 numLosers = Algorithm::Reduce(vtkm::cont::make_ArrayHandleCast<vtkm::UInt32>(isActive), 
						 (vtkm::UInt32)0);
      vtkm::UInt32 numWinners = (vtkm::UInt32) numActiveEntries - numLosers;
      std::cout << "numLosers = " << numLosers << "\n";
      std::cout << "numWinners = " << numWinners << "\n";
      std::cout << "percent found in table = " << numWinners / (1.0f*numActiveEntries) << "\n";
 
      numActiveEntries = numLosers;  
      subTableStart += subTableSize;
 
      #if 0 
      if (numActiveEntries < 100000 && 
          numActiveEntries > 0 && 
          numActiveEntries < (table_size-subTableStart))
      {
        std::cout << "remaining table slots = " << table_size-subTableStart << "\n";
        Algorithm::CopyIf(query_keys,
		          isActive,
                          tempKeys,
                          numActiveEntries);
        
        //vtkm::cont::ArrayPortalToIterators iterators(hashTable.PrepareForInput<DeviceAdapter>());
        Algorithm::BinarySearch(hashTable,
                                subTableStart,
                                table_size,
                                tempKeys, 
                                tempVals); 
      
        std::cout << "Starting CopyToSubTable\n";

        CopyToSubTable copyWorklet(subTableStart);
        vtkm::worklet::DispatcherMapField<CopyToSubTable,DeviceAdapter> copyTableDispatcher(copyWorklet);
        copyTableDispatcher.Invoke(tempKeys, tempVals, hashTable);

        debug::HashFightDebug(tempKeys, "compactedKeys");
        debug::HashFightDebug(tempVals, "compactedVals");
        debug::HashFightDebug(hashTable, "hashTable");
  
        totalSpaceUsed += numActiveEntries;

        break;
      } 
      #endif
 
      numLoops++;
      subTableSize = sub_table_starts[numLoops];
      totalSpaceUsed += subTableSize;


      std::cerr << "==================================================\n";


      std::cout << "subTableStart: " << subTableStart << std::endl;
      std::cout << "subTableSize: " << subTableSize << std::endl;
      std::cout << "numActiveEntries: " << numActiveEntries << std::endl;
      std::cout << "numLoops: " << numLoops << std::endl;


    }

    std::cout << "Total space used: " << totalSpaceUsed << "\n";
    std::cout << "Total allocated space: " << table_size << "\n";

  }
#endif


  template<typename DeviceAdapter>
  void 
  Fight(const UInt32HandleType &keys,
        const UInt32HandleType &vals,
        UInt64HandleType &hashTable,
        //std::vector<std::unique_ptr<vtkm::UInt64[]>> &hashTable,
        const vtkm::Id &num_keys,
        const vtkm::Id &table_size,
        //std::vector<vtkm::UInt32> &table_size,
        const vtkm::Float32 table_size_factor)
  { 
    typedef vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter> Algorithm;

    //Number of keys to insert into hash table
    vtkm::Id numActiveEntries = num_keys;
    
    std::cout << "numEntries: " << numActiveEntries << std::endl;
    std::cout << "sizeFactor: " << table_size_factor << std::endl;
 
    //Iter 0: 32 threads per warp --> k % 512 % (sizeFactor*numActive) % 512
    //Iter 1: ? buckets --> k % 
    //Iter 2: ? buckets --> k % 
    //If sizeFactor*numActive < 512: don't % by buckets


    const vtkm::Id threadsPerWarp = 32;
    /*
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == 0)
    {
      threadsPerWarp = prop.warpSize;
      printf("max threads per warp: %d\n", threadsPerWarp); 
    }
    */

    UInt8HandleType isActive;
    ConstBitHandleType allActive(vtkm::UInt8(1), numActiveEntries);
    Algorithm::Copy(allActive, isActive);

    //IdHandleType activeKeyIds;
    //IndexHandleType tempActiveKeyIds(numKeys);
    //Algorithm::Copy(tempActiveKeyIds, activeKeyIds);

    //debug::HashFightDebug(activeKeyIds, "activeKeyIds");

    /*
    ConstBitHandleType tempDupKey(vtkm::UInt8(0), numKeys);
    Algorithm::Copy(tempDupKey, isDuplicateKey);

    debug::HashFightDebug(isDuplicateKey, "isDuplicateKey");
    */

    /*
    ConstantHandleType initHashIters(vtkm::Id(0), numKeys);
    Algorithm::Copy(initHashIters, output_key_hash_iters);
    */ 

    /*
    vtkm::cont::ArrayHandleGroupVec<UInt32HandleType,4> vecKeys;
    vtkm::cont::ArrayHandleGroupVec<UInt32HandleType,4> vecValues; 
    vtkm::cont::ArrayHandleGroupVec<UInt64HandleType,4> vecEntries;
    */

    //UInt64HandleType entries;
    //entries.Allocate(numActiveEntries);
    //vecEntries = vtkm::cont::make_ArrayHandleGroupVec<4>(entries);
    //vecKeys = vtkm::cont::make_ArrayHandleGroupVec<4>(keys);
    //vecValues = vtkm::cont::make_ArrayHandleGroupVec<4>(vals);

    #if 0
    std::cout << "Starting GetEntry\n";
    vtkm::worklet::DispatcherMapField<GetEntry> entryDispatcher;
    //entryDispatcher.Invoke(vecKeys, vecValues, vecEntries);
    entryDispatcher.Invoke(keys, vals, entries);
    debug::HashFightDebug(entries, "entries");
    #endif

    /*
    vtkm::cont::ArrayHandleGroupVec<IdHandleType,4> vecHashes;
    vtkm::cont::ArrayHandleGroupVec<UInt8HandleType,4> vecActive;
    */

    //IdHandleType activeHashes;
    UInt32HandleType hashes;
    //UInt8HandleType isActiveHandle;
    UInt32HandleType tempKeys, tempVals;
    vtkm::Id subTableStart = 0;
    vtkm::Id subTableSize = table_size_factor * numActiveEntries;   
    //vtkm::Id subTableSize = table_size;  
    vtkm::UInt64 totalSpaceUsed = subTableSize;
    vtkm::Id numLoops = 0;

    //Begin hash fighting
    while (numActiveEntries > 0)
    {
      //subTableSize = sizeFactor * numActive;
 
      std::cerr << "=============Loop " << numLoops << " =======================\n";
      std::cerr << "numLoops = " << numLoops << std::endl;
      std::cerr << "numActiveEntries = " << numActiveEntries << std::endl;
      std::cerr << "subtableSize = " << subTableSize << std::endl;
      std::cerr << "subtableStart = " << subTableStart << std::endl;

      //Pass activeKeyIds and keys into ComputeHash
	//For an activeKeyId, grab its key
	//Compute the hash of the key
	//With fnv1a: is there coherence among hashes? NO

      //Sorting the hashes: coherence -> reassign threads to sorted hashes/keys
      //Without sorting; uncoalesced memory requests during scatter, unless divergent
      //                 requests of a warp are within the same 32B/128B block
      //What type of sort will be used?: radix sort (built-in data type and default less_than comparator)
      //If sort the keys by hash value, then we could store the keys as is
      //Then do binary search to query a key later on
      //This approach: sort the hashes and also do hash fighting until key/values placed in hash table
      //Query: hash fighting to retrieve value for key
      //Comparing two approaches: are two hash-fighting phases faster than a p-nary search?
      //Cuckoo hashing: uncoalesced hashes, but do cuckoo hashing (w/out sort) to insert keys into table
      //query -> do cuckoo hashing to find key and return value
      //Comparing this with hash fight: is sort+2*hashfight < time cuckoo hashing
      //Comparing this with other: is sort+search < time cuckoo hashing

      //Gather the active keys
      //IdPermuteHandleType activeKeys(activeKeyIds, keys);

      #if 0
      vecEntries = vtkm::cont::make_ArrayHandleGroupVec<4>(entries);
      vecHashes = vtkm::cont::make_ArrayHandleGroupVec<4>(hashes);       
      vecActive = vtkm::cont::make_ArrayHandleGroupVec<4>(isActive);       
      #endif

      std::cout << "Starting ComputeHash\n";

      #if 0
      IndexHandleType workThreads(32*numActiveEntries);
      hashes.Allocate(numActiveEntries);     
      #endif

      //Hash each key to an index/location in the hash table
      ComputeHash hashWorklet(subTableSize, subTableStart);
      vtkm::worklet::DispatcherMapField<ComputeHash,DeviceAdapter> hashDispatcher(hashWorklet);
      //hashDispatcher.Invoke(activeKeys, activeHashes);
      //hashDispatcher.Invoke(vecEntries, vecHashes, hashTable);
      //hashDispatcher.Invoke(workThreads, entries, hashes, hashTable);      
      hashDispatcher.Invoke(keys, vals, isActive, hashTable);

      debug::HashFightDebug(keys, "keys");
      debug::HashFightDebug(vals, "vals");
      //debug::HashFightDebug(hashes, "hashes");
      debug::HashFightDebug(isActive, "isActive");
      debug::HashFightDebug(hashTable, "hashTable");      
 
      #if 0
      UInt32HandleType uniqueHashes, sortedHashes;
      IdHandleType hashCounts;
      Algorithm::CopyIf(hashes, isActive, sortedHashes);
      Algorithm::Sort(sortedHashes);
      Algorithm::ReduceByKey(sortedHashes, 
			     vtkm::cont::make_ArrayHandleConstant((vtkm::Id)1, num_keys),
			     uniqueHashes,
			     hashCounts,
                             vtkm::Add());
      vtkm::Id maxNumCollisions = Algorithm::Reduce(hashCounts, vtkm::Id(0), vtkm::Maximum());
      std::cout << "Max number of collisions = " << maxNumCollisions << "\n";
      
      debug::HashFightDebug(uniqueHashes, "uniqueHashes");
      debug::HashFightDebug(hashCounts, "hashCounts");
      #endif



      #if 0
      if (numLoops == 0)
        Algorithm::SortByKey(hashes, entries);
      #endif

      //Very good hash functions distribute keys to disparate locations
      //But this scattering leads to divergence of memory accesses into hash table
      //Hash fighting is robust to collisions. Can we use a weaker hash function
      //to reduce scattering (increased collisions) and improve memory coalescing?
      //Idea1: hash keys into buckets ~size of thread blocks
	  //Pros: take advantage of coalesced writes each iter (and possibly shared memory hash-fighting)
	  //Cons: Uncoalesced writes to reorganize keys by bucket each iter (fast scan also needed)
      //Idea2: Use a randomized hash function that partially maintains coherence
	  //Pros: Avoids bucketing and slightly improves coalescing for free
	  //Cons: Hard to do...similiar keys are still spread out. Hashing based on position
	  //      in keys array won't work since the positions are unknown to queries
	  //      Also, if similar keys hash coherently, then they still need to be reorganized together prior to insertion
	  //      so that nearby threads can be assigned spatially-close hashed keys
      //Idea3: Use good hash function and assume randomness and uncoalesced writes. Then optimize downstream operations
	  //Pros: Simplest approach that doesn't try to reverse the natural effects of hashing ("random" access)
	  //Cons: Assumes the scatter is necessary, forcing CheckForMatches and gathers to be optimized much more
	  //      Be liberal with kernel fusion and removing unneeded data movements

      //Scatter: Insert the ID, i, of a key into location activeHashes[i] of the hash table
      //vtkm::worklet::DispatcherMapField<Scatter, DeviceAdapter> scatterDispatcher;
      //scatterDispatcher.Invoke(entries, hashTable);
      //scatterDispatcher.Invoke(activeHashes, activeKeyIds, hashTable);


      //Gather operation for hash indices, post-scatter
      //IdPermuteHandleType winningKeyIds(activeHashes, hashTable);
      //debug::HashFightDebug(winningKeyIds, "winningKeyIds");

      //Gather operation for vertices of current hashed face
      //NestedIdPermuteHandleType winningKeys(winningKeyIds, keys);
      //NestedIdPermuteHandleType winningKeys(winningKeyIds, hashTable);
      //debug::HashFightDebug(winningKeys, "winningKeys");

      //Check for matches
      //keys:          2 4 6 8 0 	2 4 6 8 0
      //activeKeyIds:  0 1 2 3 4	1 4 3 2 0
      //activeHashes:  7 0 7 5 1	0 1 5 7 7
      //hashTable:     1 4 _ _ _ 3 _ 2	1 4 _ _ _ 3 _ 2
      //winningKeyIds: 2 1 2 3 4 (P)	1 4 3 2 2
      //winningKeys:   6 4 6 8 0 (P)	4 0 8 6 6
      //activeKeyIds:  
      //activeKeys:    2 4 6 8 0 (P)	


      //Equivalence to cuckoo hashing insertion probe: Scatter + CheckForMatches + CopyIf
      //Hard to improve Scatter given hash function, so optimize CheckForMatches
      //Fuse permutations (gathers) with Scatter and CheckForMatches; removing the SortByKey enables this 
       
      std::cout << "Starting CheckForMatches\n";
      CheckForMatches matchesWorklet(subTableSize, subTableStart);
      vtkm::worklet::DispatcherMapField<CheckForMatches, DeviceAdapter> matchDispatcher(matchesWorklet);
      matchDispatcher.Invoke(//winningKeys, 
                             //activeKeys,
                             keys,
			     //hashes,
                             hashTable,
                             //winningKeyIds,
                             //activeKeyIds,
                             isActive);

      debug::HashFightDebug(isActive, "isActive");

      vtkm::UInt32 numLosers = Algorithm::Reduce(vtkm::cont::make_ArrayHandleCast<vtkm::UInt32>(isActive), 
						 (vtkm::UInt32)0);
      vtkm::UInt32 numWinners = (vtkm::UInt32) numActiveEntries - numLosers;
      std::cout << "numLosers = " << numLosers << "\n";
      std::cout << "numWinners = " << numWinners << "\n";
      std::cout << "percent placed in table = " << numWinners / (1.0f*numActiveEntries) << "\n";
 

      numActiveEntries = numLosers;  
      subTableStart += subTableSize;

      #if 0     
      if (numLoops < 1)
        break;      
      #endif
 
      #if 0 
      if (numActiveEntries < 100000 && 
          numActiveEntries > 0 && 
          numActiveEntries < (table_size-subTableStart))
      {
        std::cout << "remaining table slots = " << table_size-subTableStart << "\n";
        Algorithm::CopyIf(keys,
		          isActive,
                          tempKeys,
                          numActiveEntries);
        Algorithm::CopyIf(vals,
		          isActive,
                          tempVals,
                          numActiveEntries);
        Algorithm::SortByKey(tempKeys, tempVals); 
      
        std::cout << "Starting CopyToSubTable\n";

        CopyToSubTable copyWorklet(subTableStart);
        vtkm::worklet::DispatcherMapField<CopyToSubTable,DeviceAdapter> copyTableDispatcher(copyWorklet);
        copyTableDispatcher.Invoke(tempKeys, tempVals, hashTable);

        debug::HashFightDebug(tempKeys, "compactedKeys");
        debug::HashFightDebug(tempVals, "compactedVals");
        debug::HashFightDebug(hashTable, "hashTable");
  
        totalSpaceUsed += numActiveEntries;

        break;
      } 
      #endif

      /*
      std::cout << "Starting CopyIf and Copy\n";
      Algorithm::CopyIf(//activeKeyIds,
                        keys,
		        isActive,
                        outputHandle);
      Algorithm::Copy(outputHandle, keys);
      Algorithm::CopyIf(vals,
		        isActive,
                        outputHandle); 
      Algorithm::Copy(outputHandle, vals);
      outputHandle.ReleaseResourcesExecution();
      //Algorithm::Copy(outputHandle, activeKeyIds);
      //debug::HashFightDebug(activeKeyIds, "activeKeyIds");
      //debug::HashFightDebug(entries, "entries");
      */


      //outputHandle.ReleaseResources();      
      
      /*
      Algorithm::CopyIf(isInactiveKey,
                        isInactiveKey,
                        bitHandle,
                        IsIntValue(0));
      
      Algorithm::Copy(bitHandle, isInactiveKey);
      debug::HashFightDebug(isInactiveKey, "isInactiveKey");
      //bitHandle.ReleaseResources();
      */
      /*
      IdHandleType inactivePrefixSum;
      vtkm::Id numInactive = Algorithm::ScanInclusive(isInactiveKey,
                                                      inactivePrefixSum);
      numActive = numActive - numInactive;
      isInactiveKey.Shrink(numActive);
      */    

      subTableSize = table_size_factor * numActiveEntries;
      totalSpaceUsed += subTableSize;
      //subTableBuckets = vtkm::Ceil(subTableSize / threadsPerWarp);
      //hashTable.Shrink(factor*numActive);
      //vtkm::Id numInactive = prevActive - numActive;


      std::cerr << "==================================================\n";

      numLoops++;

      std::cout << "subTableStart: " << subTableStart << std::endl;
      std::cout << "subTableSize: " << subTableSize << std::endl;
      //std::cout << "subTableBuckets: " << subTableBuckets << std::endl;
      std::cout << "numActiveEntries: " << numActiveEntries << std::endl;
      std::cout << "numLoops: " << numLoops << std::endl;

      #if 1
      #endif

    }  //End of while loop
    
    std::cout << "Total space used: " << totalSpaceUsed << "\n";
    std::cout << "Total allocated space: " << table_size << "\n";

  }
 
  template <typename DeviceAdapter>
  void 
  Insert(const UInt32HandleType &input_keys,
         const UInt32HandleType &input_vals,
         UInt64HandleType &hash_table,
         //std::vector<std::unique_ptr<vtkm::UInt64[]>> &hash_table,
         const vtkm::Id &num_keys,
         //std::vector<vtkm::UInt32> &table_size,
         const vtkm::Id &table_size,
         const vtkm::Float32 &size_factor)
  {
    std::cerr << "Calling Fight function..." << std::endl;
    //Hash fight to insert keys into hash table
    //UInt8HandleType isDuplicateKey;
    Fight<DeviceAdapter>(input_keys,
		         input_vals, 
                         hash_table,
		         num_keys,
			 table_size,
                         size_factor);
  }

#if 0
  template <typename DeviceAdapter>
  void 
  Insert(const IdHandleType &input_keys,
         IdHandleType &hash_table,
         IdHandleType &output_keys)
  {
    //Hash fight to detect any duplicate input keys
    UInt8HandleType isDuplicateKey;
    Fight<DeviceAdapter>(input_keys, 
                         hash_table,
                         isDuplicateKey);
    
    //Remove the duplicate keys
    Algorithm::CopyIf(input_keys,
                      isDuplicateKey,
                      output_keys);
  }
#endif

  template<typename DeviceAdapter>
  void
  Delete(const IdHandleType &delete_keys,
              IdHandleType &hash_table)
  {
    //Do something similar to Insert here, but
    //remove the duplicate values

  }


public:

  ///////////////////////////////////////////////////
  /// \brief HashFight:
  ///                   
  /// \param shapes: Cell shape types
  /// \param numIndices: Per-cell number of indices
  /// \param conn: Cell point connectivity
  /// \param output_shapes: Output triangular external face types
  /// \param output_numIndices: Output number of indices per external face
  /// \param output_conn: Output external face point connectivity
  template <typename StorageT,
            typename DeviceAdapter>
  VTKM_CONT void Run(const vtkm::cont::ArrayHandle<vtkm::UInt32, StorageT> &input_keys,
                     const vtkm::cont::ArrayHandle<vtkm::UInt32, StorageT> &input_values,
           //const vtkm::cont::ArrayHandle<vtkm::Id, StorageT> query_keys, 
                     const vtkm::Float32 &table_size_factor,
           //const vtkm::cont::ArrayHandle<vtkm::Id, StorageT> delete_keys,
           //vtkm::cont::ArrayHandle<vtkm::Id, StorageT> &query_results,           
                     vtkm::cont::ArrayHandle<vtkm::UInt64, StorageT> &output_hash_table,
                     //std::vector<std::unique_ptr<vtkm::UInt64[]>> &output_hash_table,
		     //std::vector<vtkm::UInt32> &output_table_size,
                     DeviceAdapter
                    )
  {
    typedef vtkm::cont::DeviceAdapterAlgorithm<DeviceAdapter> Algorithm;
    
 
    //faceVertices Output: <476> <473> <463> <763> (cell 1) | <463> <462> <432> <632> (cell 2) ...
    //vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id, 3> > faceVertices;
    
    //-----------------------Begin Hashing Phase------------------//

    const vtkm::Id numKeys = input_keys.GetNumberOfValues();

    vtkm::Id tableSize = 1.575 * table_size_factor * numKeys;

    //std::cout << "tableSize = " << tableSize << std::endl;

    //tableSize = table_size_factor * numKeys;
    
    //std::cout << "tableSize = " << tableSize << std::endl;
    const vtkm::Float32 log = vtkm::Log2((vtkm::Float32)tableSize);
    const vtkm::Float32 upperLog = vtkm::Ceil(log);
    const vtkm::Float32 powerTwo = vtkm::Pow((vtkm::Float32)2, upperLog); 
    //tableSize = (vtkm::Id)powerTwo; 
    std::cout << "log = " << log << std::endl;
    std::cout << "upperLog = " << upperLog << std::endl;
    std::cout << "powerTwo = " << powerTwo << std::endl;


    vtkm::UInt32 maxVal = ~(static_cast<vtkm::UInt32>(0));
    std::cout << "max uint32 size = " << maxVal << "\n";
    std::cout << "tableSize = " << tableSize << std::endl;
    debug::HashFightDebug(input_keys, "input_keys");
    debug::HashFightDebug(input_values, "input_values");

    //Allocate the multi-level hash table
    Algorithm::Copy(vtkm::cont::make_ArrayHandleConstant(emptyEntry, tableSize), output_hash_table);
    //output_hash_table.Allocate(tableSize);
    //debug::HashFightDebug(output_hash_table, "output_hash_table");
    
    std::cout << "Calling Insert function..." << std::endl;
    //Insert keys into the hash table--build phase
    Insert<DeviceAdapter>(input_keys, input_values, output_hash_table, numKeys, tableSize, table_size_factor); 

    
    //Query<DeviceAdapter>(query_keys, output_hash_table, sizeFactor);


    //--------------End Hashing Phase-------------------------//


    //End of algorithm
  }

}; //struct HashFight


}} //namespace vtkm::worklet

#endif //vtk_m_worklet_HashFight_h
