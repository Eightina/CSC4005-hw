//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void inline assign_cuts(int total_workload, int num_tasks, int* cuts) {
    int work_num_per_task = total_workload / num_tasks;
    int left_pixel_num = total_workload % num_tasks;

    int divided_left = 0;

    for (int i = 0; i < num_tasks; i++) {
        if (divided_left < left_pixel_num) {
            cuts[i+1] = cuts[i] + work_num_per_task + 1;
            divided_left++;
        } else cuts[i+1] = cuts[i] + work_num_per_task;
    }
}

inline int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

inline void quickSortKernel(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSortKernel(vec, low, pivotIndex - 1);
        quickSortKernel(vec, pivotIndex + 1, high);
    }
}

// inline void 

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status, int cuts[]) {
    quickSortKernel(vec, cuts[taskid], cuts[taskid + 1]);
    if (taskid == MASTER) {
        int *cur_res = malloc(sizeof(int) * (cuts[taskid + 1] - cuts[taskid])); 
    } else {

    }

}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
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

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    // job patition
    int cuts[numtasks + 1];
    assign_cuts(size, numtasks, cuts);

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status, cuts);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}