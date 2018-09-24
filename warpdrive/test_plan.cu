#include <iostream>
#include <unordered_map>

#include "../include/warpdrive.cuh"
#include "../tools/binary_io.h"


dim3 ComputeGridDim(uint64_t n, uint64_t blockSize) {
    // Round up in order to make sure all items are hashed in.
    dim3 grid( (n + blockSize-1) / blockSize );
    /*
    if (grid.x > kGridSize) {
        grid.y = (grid.x + kGridSize - 1) / kGridSize;
        grid.x = kGridSize;
    }
    */
    return grid;
}

int CheckResults_basic(const uint64_t kInputSize,
                       const std::unordered_map<unsigned, unsigned> &pairs,
                       const warpdrive::policies::PackedPairDataPolicy<>::data_t *query_entries)
{
  using index_t = uint64_t;
  index_t errors = 0;
  for (index_t i = 0; i < kInputSize; i++)
  {
    unsigned actual_value = warpdrive::policies::PackedPairDataPolicy<>::nop_op::identity;
    std::unordered_map<unsigned, unsigned>::const_iterator it =
            pairs.find(query_entries[i].get_key());
    if (it != pairs.end())
      actual_value = it->second;
    if (actual_value != query_entries[i].get_value())
    {
      errors++;
      printf("\t\t\tError for key %10u: Actual value is "
                   "%10u, but hash returned %10u.\n",
                   query_entries[i].get_key(), actual_value, query_entries[i].get_value());
    }
  }
  return errors;
}


int main(int argc, char const *argv[]) {

    using namespace warpdrive;
    using namespace std;

    if (argc < 4)
    {
        cerr << "ERROR: Not enough parameters (read \"PARAMS\" section inside the main function)" << endl;
        return -1;
    }

    //the global index type to use
    using index_t = uint64_t;

    //PARAMS
    //the size of the thread groups (must be available at compile time)
    static constexpr index_t group_size = 32;
    //output verbosity (must be available at compile time)
    static constexpr index_t verbosity = 2;
    //filename for test data (dumped with binary_io.h)
    const string  filename = argv[1];
    //length of test data
    const index_t len_data = atoi(argv[2]);
    //load factor of the hash table
    const float   load     = atof(argv[3]);
    //capacity of the hash table
    const index_t capacity = len_data/load;
    //max chaotic probing attempts
    const index_t lvl1_max = 100000;
    //max linear probing attempts
    const index_t lvl2_max = 32/group_size;
    //number of threads per CUDA block (must be multiple of group_size)
    const index_t threads_per_block = 256; 
    //number of CUDA blocks per grid  
    const index_t blocks_per_grid = (1UL << 31)-1;
    //const index_t blocks_per_grid = 120;
    //const index_t blocks_per_grid = ComputeGridDim(len_data, threads_per_block).x;
    //const index_t blocks_per_grid = (atoi(argv[6]) != 0) ? atoi(argv[6]): (1UL << 31)-1;
    //id of selected CUDA device
    const index_t device_id = 0;

    if (verbosity > 0)
    {
        cout << "================= WARPDRIVE ================" << endl;
        cout << "================== PARAMS =================="
             << "\n(static) group_size=" << group_size
             << "\n(static) verbosity=" << verbosity
             << "\nfilename=" << filename
             << "\nlen_data=" << len_data
             << "\nload=" << load
             << "\ncapacity=" << capacity
             << "\nlvl1_max=" << lvl1_max
             << "\nlvl2_max=" << lvl2_max
             << "\nblocks_per_grid=" << blocks_per_grid
             << "\nthreads_per_block=" << threads_per_block
             << "\ndevice_id=" << device_id
             << endl;
    }

    //DECLS
    //data policy
    using data_p    = warpdrive::policies::PackedPairDataPolicy<>;
    //failure policy
    using failure_p = std::conditional<(verbosity > 1), //if verbosity < 2 ignore failures
                                       warpdrive::policies::PrintIdFailurePolicy,
                                       warpdrive::policies::IgnoreFailurePolicy>::type;

   //data types
   using data_t  = data_p::data_t;
   using key_t   = data_p::key_t;
   using value_t = data_p::value_t;

    //plan (the meat)
    using plan_t = plans::BasicPlan<group_size, //size of parallel probing group
                                    data_p,
                                    failure_p,
                                    index_t>; //index type to use>

    //config struct (probing lengths and kernel launch config)
    plan_t::config_t config(lvl1_max,
                            lvl2_max,
                            blocks_per_grid,
                            threads_per_block);

    //set the selected CUDA device
    cudaSetDevice(device_id); CUERR

    //load random keys
    std::cout << "Loading input keys...\n";
    key_t * keys_h = new key_t[len_data];
    load_binary<key_t>(keys_h, len_data, filename+"/inputKeys-"+std::to_string(len_data)+"-0");

    //load random values 
    std::cout << "Loading input values...\n";
    value_t * vals_h = new value_t[len_data];
    load_binary<value_t>(vals_h, len_data, filename+"/inputVals-"+std::to_string(len_data)+"-0");

    #if 0
    //Save the original input for checking the results.
    std::cout << "Saving key-val pairs...\n";
    std::unordered_map<key_t, value_t> pairs_basic;
    for (index_t i = 0; i < len_data; i++)
    {
      pairs_basic[keys_h[i]] = vals_h[i];
    }
    #endif

    //the hash table
    std::cout << "Allocating the hash table on device...\n";
    data_t * hash_table_d; cudaMalloc(&hash_table_d, sizeof(data_t)*capacity); CUERR

    //test data
    std::cout << "Allocating the key-val pairs on device...\n";
    data_t * data_h = new data_t[len_data];
    data_t * data_d; cudaMalloc(&data_d, sizeof(data_t)*len_data); CUERR

    //TESTS/BENCHMARKS
    if (verbosity > 0)
    {
        cout << "============= TESTS/BENCHMARK =============" << endl;
    }

    //init failure handler
    failure_p failure_handler = failure_p();
    failure_handler.init();

    //the first task to execute
    using elem_op_1 = data_p::update_op;
    static constexpr auto table_op_1 = plan_t::table_op_t::insert;

    //the second task to execute
    using elem_op_2 = data_p::nop_op;
    static constexpr auto table_op_2 = plan_t::table_op_t::retrieve;

    //init hash table
    memset_kernel
    <<<SDIV(capacity, 1024), 1024>>>
    (hash_table_d, capacity, data_t(data_p::empty_key, elem_op_1::identity));

    //init input data
    #pragma omp parallel for
    for (index_t i = 0; i < len_data; i++)
    {
        data_h[i] = data_t(keys_h[i], vals_h[i]);
    }

    cudaMemcpy(data_d, data_h, sizeof(data_t)*len_data, H2D); CUERR

    //execute task
    std::cout << "Inserting keys into hash table...\n";
    TIMERSTART(op1)
    plan_t::table_operation<table_op_1,
                            elem_op_1>
    (data_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
    TIMERSTOP(op1)

    //Load the random query keys
    std::cout << "Loading the query keys...\n";
    key_t * query_keys_h = new key_t[len_data];
    load_binary<key_t>(query_keys_h, len_data, filename+"/queryKeys-"+std::to_string(len_data)+"0-10-0");
    
    //Set the default empty values for the query keys 
    #pragma omp parallel for
    for (index_t i = 0; i < len_data; i++)
    {
        data_h[i] = data_t(query_keys_h[i], elem_op_2::identity);
        //data_h[i].set_value(elem_op_2::identity);
    }

    cudaMemcpy(data_d, data_h, sizeof(data_t)*len_data, H2D); CUERR

    //retrieve results
    std::cout << "Query the hash table...\n";
    TIMERSTART(op2)
    plan_t::table_operation<table_op_2,
                            elem_op_2>
    (data_d, len_data, hash_table_d, capacity, failure_handler, 0, config);
    TIMERSTOP(op2)

    cudaMemcpy(data_h, data_d, sizeof(data_t)*len_data, D2H); CUERR

    #if 0
    index_t errors = CheckResults_basic(len_data,
                                        pairs_basic,
                                        data_h);
    if (errors > 0)
      printf("%lu errors found\n", errors);
    else
      printf("No errors found, test passes\n");
    #endif

    //free memory
    delete[] keys_h;
    delete[] data_h;

    cudaFree(hash_table_d);
    cudaFree(data_d);
}
