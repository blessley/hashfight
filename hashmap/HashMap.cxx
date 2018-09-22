#include "tbb/concurrent_hash_map.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"

#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>

//Signifies that a query key was not found.
constexpr unsigned int keyNotFound = 0xffffffffu;

unsigned int CheckResults_basic(unsigned int kInputSize,
                       const std::unordered_map<unsigned int, unsigned int> &pairs,
                       const unsigned int *query_keys,
                       const unsigned int *query_vals)
{
  unsigned int errors = 0;
  for (unsigned i = 0; i < kInputSize; ++i)
  {
    unsigned actual_value = keyNotFound;
    std::unordered_map<unsigned int, unsigned int>::const_iterator it =
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

template <typename T>
void load_binary(const T * data,
                 const size_t length,
                 const std::string& filename)
{
    std::ifstream ifile(filename, std::ios::binary);
    ifile.read((char*) data, sizeof(T)*length);
    ifile.close();
}


inline size_t MurmurHash3(const unsigned int &x)
{
  unsigned int h = x ^ (x >> 16);
  h *= 0x7feb352d;
  h ^= h >> 15;
  h *= 0x846ca68b;
  h ^= h >> 16;
  return (size_t)h;
}

//Structure that defines hashing and comparison operations for key type.
struct MyHashCompare 
{
  static size_t hash(const unsigned int &x) 
  {
    return MurmurHash3(x);
  }
    
  //True if keys are equal
  static bool equal(const unsigned int &x, const unsigned int &y) 
  {
    return x == y;
  }
};


int main(int argc, char** argv)
{


  const std::string data_dir(argv[6]);
  double insertTime, queryTime;
  tbb::tick_count start, stop;

  std::cout << "========================TBB Concurrent HashMap"
            << "==============================\n";


  unsigned int kInputSize = (unsigned int)std::atoi(argv[1]);
  float loadFactor = (float)std::atof(argv[2]);
  unsigned int* input_keys = new unsigned int[kInputSize];
  unsigned int* input_vals = new unsigned int[kInputSize];
  unsigned int* query_keys = new unsigned int[kInputSize];
  unsigned int* query_vals = new unsigned int[kInputSize];
 
  memset(query_vals, keyNotFound, kInputSize*sizeof(unsigned int));
 
  std::cout << "Loading binary of input keys...\n";
  load_binary(input_keys, kInputSize, data_dir + "/inputKeys-" + std::string(argv[1]) + "-" + std::string(argv[5])); 

  std::cout << "Loading binary of input vals...\n";
  load_binary(input_vals, kInputSize, data_dir + "/inputVals-" + std::string(argv[1]) + "-" + std::string(argv[5])); 

  std::cout << "Loading binary of query keys...\n";
  load_binary(query_keys, kInputSize, data_dir + "/queryKeys-" + std::string(argv[1]) + "-" + std::string(argv[3]) + "-" + std::string(argv[4]) + "-" + std::string(argv[5])); 

#if 0
  std::unordered_map<unsigned int, unsigned int> pairs_basic;
  for (int i = 0; i < kInputSize; i++)
  {
    pairs_basic[input_keys[i]] = input_vals[i];
  }
#endif

  std::cout << "Inserting pairs...\n";
  using HashTable = tbb::concurrent_hash_map<unsigned int, unsigned int, MyHashCompare>;
  HashTable table(kInputSize*loadFactor);
  tbb::auto_partitioner ap;
  start = tbb::tick_count::now();
  tbb::parallel_for (
    tbb::blocked_range<unsigned int>(0, kInputSize),
    [&](tbb::blocked_range<unsigned int> r)
	{
          HashTable::accessor a;
	  for (auto i = r.begin(); i != r.end(); ++i)
	  {
            table.insert(a, input_keys[i]);
            a->second = input_vals[i];
            a.release();
          }
        }
     , ap
  );  
  stop = tbb::tick_count::now();
  insertTime = (stop - start).seconds();
  //std::cout << "Sort: elapsed : " << sortTime/1000 << "\n";
  

  std::cout << "Querying keys...\n";
  start = tbb::tick_count::now();
  tbb::parallel_for (
    tbb::blocked_range<unsigned int>(0, kInputSize),
    [&](tbb::blocked_range<unsigned int> r)
	{
          HashTable::const_accessor a;
	  for (auto i = r.begin(); i != r.end(); ++i)
	  {
            if (table.find(a, query_keys[i]))
              query_vals[i] = a->second;
            a.release();
          }
        }
     , ap
  );
  stop = tbb::tick_count::now();
  queryTime = (stop - start).seconds();

#if 0
  unsigned int errors = CheckResults_basic(kInputSize,
                                  pairs_basic,
                                  query_keys,
                                  query_vals);
  if (errors > 0)
    printf("%u errors found\n", errors);
  else
    printf("No errors found, test passes\n");
#endif

 
  delete [] query_keys;
  delete [] query_vals;

  //std::cout << "Search: elapsed : " << searchTime/1000 << "\n";
  std::cout << insertTime << "," << queryTime << "\n";

  delete [] input_keys;
  delete [] input_vals;

  return 0;
}
