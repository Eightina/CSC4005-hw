// //
// // Created by Yang Yufan on 2023/10/31.
// // Email: yufanyang1@link.cuhk.edu.cn
// //
// // Parallel Bucket Sort with MPI
// //

// #include <iostream>
// #include <vector>
// #include <mpi.h>
// #include "../utils.hpp"
// #include "string.h"

// #define MASTER 0
// #define TAG_GATHER 0

// int inline lenSum(int* array, int len) {
//     int res = 0;
//     for (int i = 0; i < len; ++i) {
//         res+=array[i];
//     }
//     return res;
// }
// int partition(int* vec, int low, int high) {
//     int pivot = vec[high];
//     int i = low - 1;

//     for (int j = low; j < high; j++) {
//         if (vec[j] <= pivot) {
//             i++;
//             std::swap(vec[i], vec[j]);
//             // int temp = vec[i];
//             // vec[i] = vec[j];
//             // vec[j] = temp;
//         }
//     }

//     std::swap(vec[i + 1], vec[high]);
//     return i + 1;
// }
// void quickSort(int* vec, int low, int high) {
//     if (low < high) {
//         int pivotIndex = partition(vec, low, high);
//         quickSort(vec, low, pivotIndex - 1);
//         quickSort(vec, pivotIndex + 1, high);
//     }
// }
// int partition(std::vector<int> &vec, int low, int high) {
//     int pivot = vec[high];
//     int i = low - 1;

//     for (int j = low; j < high; j++) {
//         if (vec[j] <= pivot) {
//             i++;
//             std::swap(vec[i], vec[j]);
//         }
//     }

//     std::swap(vec[i + 1], vec[high]);
//     return i + 1;
// }
// void quickSort(std::vector<int> &vec, int low, int high) {
//     if (low < high) {
//         int pivotIndex = partition(vec, low, high);
//         quickSort(vec, low, pivotIndex - 1);
//         quickSort(vec, pivotIndex + 1, high);
//     }
// }

// void insertionSort(std::vector<int>& bucket) {
//     for (int i = 1; i < bucket.size(); ++i) {
//         int key = bucket[i];
//         int j = i - 1;

//         while (j >= 0 && bucket[j] > key) {
//             bucket[j + 1] = bucket[j];
//             j--;
//         }

//         bucket[j + 1] = key;
//     }
// }

// void insertionSort(int* bucket, int bucketLen) {
//     for (int i = 1; i < bucketLen; ++i) {
//         int key = bucket[i];
//         int j = i - 1;

//         while (j >= 0 && bucket[j] > key) {
//             bucket[j + 1] = bucket[j];
//             j--;
//         }

//         bucket[j + 1] = key;
//     }
// }

// std::vector<std::vector<int>>& bucketSortKernel(int max_val, int min_val, std::vector<int>& vec, int num_buckets) {
//     int range = max_val - min_val + 1;
//     int small_bucket_size = range / num_buckets;
//     int large_bucket_size = small_bucket_size + 1;
//     int large_bucket_num = range - small_bucket_size * num_buckets;
//     int boundary = min_val + large_bucket_num * large_bucket_size;

//     std::vector<std::vector<int>>* buckets = new std::vector<std::vector<int>>(num_buckets);
//     // Pre-allocate space to avoid re-allocation
//     for (std::vector<int>& bucket : *buckets) {
//         bucket.reserve(large_bucket_num);
//     }

//     // Place each element in the appropriate bucket
//     for (int num : vec) {
//         int index;
//         if (num < boundary) {
//             index = (num - min_val) / large_bucket_size;
//         } else {
//             index = large_bucket_num + (num - boundary) / small_bucket_size;
//         }
//         if (index >= num_buckets) {
//             // Handle elements at the upper bound
//             index = num_buckets - 1;
//         }
//         (*buckets)[index].push_back(num);
//     }

//     // Sort each bucket using insertion sort
//     for (std::vector<int>& bucket : *buckets) {
//         // insertionSort(bucket);
//         quickSort(bucket, 0, bucket.size()-1);
//     }
//     return *buckets;
//     // Combine sorted buckets to get the final sorted array
//     // int index = 0;
//     // for (const std::vector<int>& bucket : buckets) {
//     //     for (int num : bucket) {
//     //         vec[index++] = num;
//     //     }
//     // }
// }

// void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status, std::vector<int> cuts) {
//     // attention here, buckets should be split as the full vector
//     int start = cuts[taskid], end = cuts[taskid + 1] - 1, afterEnd = cuts[taskid + 1];
//     std::vector<int>* cur_vec = new std::vector<int>(vec.begin() + start, vec.begin() + afterEnd);
//     int max_val = *std::max_element(vec.begin(), vec.end());
//     int min_val = *std::min_element(vec.begin(), vec.end());
//     std::vector<std::vector<int>> cur_buckets = bucketSortKernel(max_val, min_val, *cur_vec, num_buckets); 
    
//     // communicate each others' buckets sizes
//     int* send_counts = (int*)malloc(sizeof(int) * num_buckets);
//     int* sdispls = (int*)malloc(sizeof(int) * num_buckets);
//     int* cur_buckets_sizes = (int*)malloc(sizeof(int) * num_buckets);
//     int* i_buckets_sizes = (int*)malloc(sizeof(int) * num_buckets);
//     for (int i = 0; i < num_buckets; ++i) {
//         send_counts[i] = 1;
//         sdispls[i] = i;
//         cur_buckets_sizes[i] = cur_buckets[i].size();
//     }
    
//     printf("%d init\n", taskid);
//     MPI_Alltoallv(cur_buckets_sizes, send_counts, sdispls, MPI_INT,
//                     i_buckets_sizes, send_counts, sdispls, MPI_INT, 
//                     MPI_COMM_WORLD
//                 );
//     printf("%d buckets sizes comm\n", taskid);
//     // MPI_Barrier(MPI_COMM_WORLD); // sync
    
//     // send buckets as pre-assumed
//     int* i_buckets = (int*)malloc(sizeof(int) * lenSum(i_buckets_sizes, num_buckets));
//     int* rdispls = (int*)malloc(sizeof(int) * num_buckets);
//     for (int i = 0; i < num_buckets; ++i) {
//         if (i == 0) {
//             sdispls[i] = 0;
//             rdispls[i] = 0;
//             continue;
//         }
//         sdispls[i] = cur_buckets_sizes[i - 1] + sdispls[i - 1];
//         rdispls[i] = i_buckets_sizes[i - 1] + rdispls[i - 1];
//     }
//     int* cur_buckets_send = (int*)malloc(sizeof(int) * (cuts[taskid + 1] - cuts[taskid]));
//     for (int i = 0; i < num_buckets; ++i) {
//         memcpy(cur_buckets_send + sdispls[i], cur_buckets[i].data(), cur_buckets_sizes[i] * sizeof(int));
//         // printf("%d, %d, %d\n",sdispls[i], cur_buckets[i].data()[3], cur_buckets_sizes[i]);
//     }
    
//     MPI_Alltoallv(cur_buckets_send, cur_buckets_sizes, sdispls, MPI_INT,
//                     i_buckets, i_buckets_sizes, rdispls, MPI_INT, 
//                     MPI_COMM_WORLD
//                 );
//     printf("%d i buckets comm\n", taskid);
//     // MPI_Barrier(MPI_COMM_WORLD); // sync
    
//     // sort the gathered i buckets  
//     int i_buckets_total_len = lenSum(i_buckets_sizes, num_buckets);
//     // insertionSort(i_buckets, i_buckets_total_len);
//     quickSort(i_buckets, 0, i_buckets_total_len - 1);
//     printf("%d i buckets sorted\n", taskid);

//     // master gather all i buckets from slaves
//     MPI_Barrier(MPI_COMM_WORLD); // sync
//     MPI_Status recv_status;
//     int is_sent;
//     int finished_num = 0;
//     int* recv_t = vec.data();

//     if (taskid == MASTER) {
//         memcpy(recv_t, i_buckets, sizeof(int) * i_buckets_total_len);
//         recv_t += i_buckets_total_len;
//         ++finished_num;
//         while (finished_num < numtasks) {
//             // MPI_Recv(vec.data() + rdispls[i], )
//             MPI_Iprobe(MPI_ANY_SOURCE, TAG_GATHER, MPI_COMM_WORLD, &is_sent, &recv_status);
//             if (!is_sent) {
//                 continue;
//             }
//             int recv_length = 0;
//             MPI_Get_count(&recv_status, MPI_INT, &recv_length);
//             MPI_Recv(recv_t, recv_length, MPI_INT, recv_status.MPI_SOURCE, TAG_GATHER, MPI_COMM_WORLD, &recv_status);
//             printf("%d received\n", recv_status.MPI_SOURCE);
//             recv_t += recv_length;
//             ++finished_num;            
//         }
//     } else {
//         MPI_Send(i_buckets, i_buckets_total_len, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
//         printf("%d sent\n", taskid);
//     }

//     // release heap memory
//     delete cur_vec;
//     delete []cur_buckets_sizes;
//     delete []i_buckets_sizes;
//     delete []i_buckets;
//     delete []cur_buckets_send;
//     delete []send_counts;
//     delete []sdispls;
//     delete []rdispls;

// }

// int main(int argc, char** argv) {
//     // Verify input argument format
//     if (argc != 3) {
//         throw std::invalid_argument(
//             "Invalid argument, should be: ./executable vector_size bucket_num\n"
//             );
//     }
//     // Start the MPI
//     MPI_Init(&argc, &argv);
//     // How many processes are running
//     int numtasks;
//     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
//     // What's my rank?
//     int taskid;
//     MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
//     // Which node am I running on?
//     int len;
//     char hostname[MPI_MAX_PROCESSOR_NAME];
//     MPI_Get_processor_name(hostname, &len);
//     MPI_Status status;

//     const int size = atoi(argv[1]);

//     const int bucket_num = atoi(argv[2]);

//     const int seed = 4005;

//     std::vector<int> vec = createRandomVec(size, seed);
//     std::vector<int> vec_clone = vec;

//     // job patition
//     std::vector<int> cuts = createCuts(0, vec.size() - 1, numtasks);
    
//     auto start_time = std::chrono::high_resolution_clock::now();

//     bucketSort(vec, bucket_num, numtasks, taskid, &status, cuts);

//     if (taskid == MASTER) {
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//             end_time - start_time);
        
//         std::cout << "Bucket Sort Complete!" << std::endl;
//         std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
//                 << std::endl;
        
//         checkSortResult(vec_clone, vec);
//     }

//     MPI_Finalize();
//     return 0;
// }