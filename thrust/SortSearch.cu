

#include <thrust/device_vector.h>
#include <thrust/sort.h>



int main(int argc, char** argv)
{






  std::cout << "========================Thrust Sort+BinarySearch"
            << "==============================\n";

  
  thrust::device_vector<unsigned int> thrust_test_keys(input_keys,
					               input_keys + kInputSize);

  thrust::sort(thrust_test_keys.begin(), thrust_test_keys.end());


















}
