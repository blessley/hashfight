#include <mt19937ar.h>
#include "random_numbers.h"

#include <string.h>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

  
int main(int argc, char** argv)
{
  if (argc < 2)
    return -1;


  unsigned int* input_keys = NULL;
  unsigned int* input_vals = NULL;
  unsigned *query_keys = NULL;
  unsigned int pool_size = 0;
  unsigned int* number_pool = NULL;
 
  int num_runs = std::stoi(argv[1]); 
  const int failure_trials = 10; 
  float failure_rate = 0.0f;
  const unsigned int million = 1000000;
  const unsigned int maxInputSize = 500 * million;
  const unsigned int minInputSize = 500 * million;
  const unsigned int inputStepSize = minInputSize;
  std::string data_dir(argv[2]);
  //const int numSpaceUsagesToTest = 9;
  //const float kSpaceUsagesToTest[9] = {1.03f, 1.05f, 1.10f, 1.15f, 1.25f, 1.5f, 1.75f, 1.9f, 2.0f};
  for (int r = 0; r < num_runs; r++)
  {
    std::cout << "---------------RunId  " << r << "-----------------\n";
    for (unsigned int kInputSize = minInputSize; kInputSize <= maxInputSize; kInputSize += inputStepSize)
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
      dump_binary(input_keys, kInputSize, data_dir + "/inputKeys-" + std::to_string(kInputSize) + 
     					"-" + std::to_string(r));

      std::cout << "Dumping binary of input vals\n";
      dump_binary(input_vals, kInputSize, data_dir + "/inputVals-" + std::to_string(kInputSize) +   
 				      "-" + std::to_string(r));
     
      //Randomly-generate the query keys
      for (int failure = 9; failure < failure_trials; failure++)
      {
         failure_rate = failure / (float)failure_trials;
         std::cout << "Failure Rate = " << failure_rate << "\n";
         query_keys = new unsigned int[kInputSize];
         std::cout << "Generating random query keys\n";
         GenerateQueries(kInputSize, failure_rate, number_pool, query_keys); 
         std::cout << "Dumping binary of query keys\n";
         dump_binary(query_keys, kInputSize, data_dir + "/queryKeys-" + std::to_string(kInputSize) +
 				            "-" + std::to_string(failure) + "-" +
				            std::to_string(failure_trials) + "-" +
				            std::to_string(r));
         delete [] query_keys;
       } 

       delete [] number_pool;
       delete [] input_keys;
       delete [] input_vals;
     }
   } 
}


