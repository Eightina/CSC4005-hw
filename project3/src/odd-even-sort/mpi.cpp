//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void oddEvenSortKernel(std::vector<int>& vec, int low, int high, int taskid, int numtasks) {
    int trueEnd = vec.size() - 1; 
    bool sorted = false;

    int cnt = 0;
    while (!sorted) {
        sorted = true;

        // Perform the odd phase
        // for (int i = 1; i < vec.size() - 1; i += 2) {
        MPI_Isend(vec.data(), 1, MPI_INT, taskid - 1);
        for (int i = low + 1; i <= high; i += 2) { 
            if (i == high && i < trueEnd) {
                int next;

                break;
            }

            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // Perform the even phase
        // for (int i = 0; i < vec.size() - 1; i += 2) {
        for (int i = low; i <= high; i += 2) {
            if (i == high && i < trueEnd) {

                break;
            }
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }
    }
}

void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status, std::vector<int>& cuts) {


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
    std::vector<int> cuts = createCuts(0, vec.size() - 1, numtasks);

    auto start_time = std::chrono::high_resolution_clock::now();

    oddEvenSort(vec, numtasks, taskid, &status, cuts);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}