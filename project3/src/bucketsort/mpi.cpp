//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"
#include "string.h"

#define MASTER 0

int inline sum(int* array, int len) {
    int res = 0;
    for (int i = 0; i < len; ++i) {
        res+=array[i];
    }
    return res;
}

void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

std::vector<std::vector<int>>& bucketSortKernel(std::vector<int>& vec, int num_buckets) {
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = min_val + large_bucket_num * large_bucket_size;

    std::vector<std::vector<int>>* buckets = new std::vector<std::vector<int>>(num_buckets);
    // Pre-allocate space to avoid re-allocation
    for (std::vector<int>& bucket : *buckets) {
        bucket.reserve(large_bucket_num);
    }

    // Place each element in the appropriate bucket
    for (int num : vec) {
        int index;
        if (num < boundary) {
            index = (num - min_val) / large_bucket_size;
        } else {
            index = large_bucket_num + (num - boundary) / small_bucket_size;
        }
        if (index >= num_buckets) {
            // Handle elements at the upper bound
            index = num_buckets - 1;
        }
        (*buckets)[index].push_back(num);
    }

    // Sort each bucket using insertion sort
    for (std::vector<int>& bucket : *buckets) {
        insertionSort(bucket);
    }
    return *buckets;
    // Combine sorted buckets to get the final sorted array
    // int index = 0;
    // for (const std::vector<int>& bucket : buckets) {
    //     for (int num : bucket) {
    //         vec[index++] = num;
    //     }
    // }
}

void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status, std::vector<int> cuts) {
    int start = cuts[taskid], end = cuts[taskid + 1] - 1, afterEnd = cuts[taskid];
    std::vector<int>* cur_vec = new std::vector<int>(vec.begin() + start, vec.begin() + afterEnd);
    std::vector<std::vector<int>> cur_buckets = bucketSortKernel(*cur_vec, num_buckets);
    
    // communicate each others' buckets sizes
    int* cur_buckets_sizes = (int*)malloc(sizeof(int) * num_buckets);
    int* i_buckets_sizes = (int*)malloc(sizeof(int) * num_buckets);

    int* send_counts = (int*)malloc(sizeof(int) * num_buckets);
    memset(send_counts, num_buckets, 1);
    int* sdispls = (int*)malloc(sizeof(int) * num_buckets);
    for (int i = 0; i < num_buckets; ++i) sdispls[i] = i;
    for (int i = 0; i < num_buckets; ++i) {
        cur_buckets_sizes[i] = cur_buckets[i].size();
    }

    MPI_Alltoallv(cur_buckets_sizes, send_counts, sdispls, MPI_INT,
                    i_buckets_sizes, send_counts, sdispls, MPI_INT, 
                    MPI_COMM_WORLD
                );
    // delete []cur_buckets_sizes;
    // delete []send_counts;
    // delete []sdispls;
    
    // send buckets as pre-assumed
    int* i_buckets = (int*)malloc(sizeof(int) * sum(i_buckets_sizes, num_buckets));
    int* cur_buckets_send = (int*)malloc(sizeof(int) * (cuts[taskid + 1] - cuts[taskid]));
    int cpyidx = 0;
    for (int i = 0; i < num_buckets; ++i) {
        memcpy(cur_buckets_send + cpyidx, cur_buckets[i].data(), cur_buckets[i].size());
    }
    int* rdispls = (int*)malloc(sizeof(int) * num_buckets);
    for (int i = 0; i < num_buckets; ++i) {
        if (i == 0) {
            sdispls[i] = 0;
            rdispls[i] = 0;
            continue;
        }
        sdispls[i] = cur_buckets_sizes[i-1];
        rdispls[i] = i_buckets_sizes[i-1];
    }
    
    MPI_Alltoallv(cur_buckets_send, cur_buckets_sizes, sdispls, MPI_INT,
                    i_buckets, i_buckets_sizes, rdispls, MPI_INT, 
                    MPI_COMM_WORLD
                );

    // for (std::vector<int> bucket : cur_buckets) {

    // std::vector<int>* cur_vec = new std::vector<int>(vec.begin(), vec.begin());
    // std::vector<int> cur_ve(vec.begin(), vec.end());

    /* Your code here!
       Implement parallel bucket sort with MPI
    */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
            );
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const int size = atoi(argv[1]);

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    // job patition
    std::vector<int> cuts = createCuts(0, vec.size() - 1, numtasks);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status, cuts);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}