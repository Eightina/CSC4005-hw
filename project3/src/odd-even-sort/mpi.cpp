
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
#include "string.h"

#define MASTER 0
#define TAG_GATHER 0
#define TAG_RES 1

inline void oddPhase(int low, int high, std::vector<int>& vec, bool& sorted) {
    for (int i = low + 1; i < high; i += 2) { 
        if (vec[i] > vec[i + 1]) {
            std::swap(vec[i], vec[i + 1]);
            sorted = false;
        }
    }
}
inline void evenPhase(int low, int high, std::vector<int>& vec, bool& sorted) {
    for (int i = low; i < high; i += 2) {
        if (vec[i] > vec[i + 1]) {
            std::swap(vec[i], vec[i + 1]);
            sorted = false;
        }
    }
}

inline void tPhase(std::vector<int>& vec, int low, int high, 
                int taskid, int numtasks, std::vector<int>& odd_even_cuts, bool& sorted) {
    MPI_Request request;
    MPI_Status status;
    int recvNum = 0;
    if (taskid != MASTER)
        MPI_Isend(vec.data() + low, 1, MPI_INT, taskid - 1, TAG_GATHER, MPI_COMM_WORLD, &request);
    if (taskid != numtasks - 1)
        MPI_Recv(&recvNum, 1, MPI_INT, taskid + 1, TAG_GATHER, MPI_COMM_WORLD, &status);

    if (taskid != numtasks - 1) {
        if (vec[high] > recvNum) {
            MPI_Isend(vec.data() + high, 1, MPI_INT, taskid + 1, TAG_RES, MPI_COMM_WORLD, &request);
            vec[high] = recvNum;
            sorted = false;
        } else {
            MPI_Isend(&recvNum, 1, MPI_INT, taskid + 1, TAG_RES, MPI_COMM_WORLD, &request);
        }
    }

    if (taskid != MASTER) {
        MPI_Recv(vec.data() + low, 1, MPI_INT, taskid - 1, TAG_RES, MPI_COMM_WORLD, &status);
    }
    // if (taskid == MASTER)
    //     evenPhase(low, high - 1, vec, sorted);
    // else if (taskid == numtasks - 1)
    //     evenPhase(low + 1, high, vec, sorted);
    // else
    //     evenPhase(low + 1, high - 1, vec, sorted);
}

inline void ntPhase (std::vector<int>& vec, int low, int high, 
                int taskid, int numtasks, std::vector<int>& odd_even_cuts, bool& sorted) {
    oddPhase(low, high, vec, sorted);
    // MPI_Barrier(MPI_COMM_WORLD);
    evenPhase(low, high, vec, sorted);
}


// void oddEvenSortKernel(std::vector<int>& vec, int low, int high, int taskid, int numtasks, std::vector<int>& odd_even_cuts) {
//     int trueEnd = vec.size() - 1; 
//     bool sorted = false;

//     int cnt = 0;
//     int phase_flag = 0;
//     MPI_Request request;
//     MPI_Status status;

//     while (!sorted) {
//         sorted = true;

//         // Perform the odd phase
//         // for (int i = 1; i < vec.size() - 1; i += 2) {
//         phase_flag = 0;
//         if ((taskid != MASTER) && (odd_even_cuts[taskid - 1] != phase_flag))
//             MPI_Isend(vec.data(), 1, MPI_INT, taskid - 1, cnt, MPI_COMM_WORLD, &request);
//         int recvNum = 0;
//         if ((taskid != numtasks - 1) && (odd_even_cuts[taskid] != phase_flag))
//             MPI_Recv(&recvNum, 1, MPI_INT, taskid + 1, cnt, MPI_COMM_WORLD, &status);
//         oddPhase(low, high, vec, sorted);
//         MPI_Barrier(MPI_COMM_WORLD);

//         phase_flag = 1;
//         // Perform the even phase
//         // for (int i = 0; i < vec.size() - 1; i += 2) {
//         if ((taskid != MASTER) && (odd_even_cuts[taskid - 1] != phase_flag))
//             MPI_Isend(vec.data(), 1, MPI_INT, taskid - 1, cnt, MPI_COMM_WORLD, &request);
//         evenPhase(low, high, vec, sorted);
//         MPI_Barrier(MPI_COMM_WORLD);

//     }
// }

void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status, std::vector<int>& cuts) {
    std::vector<int> odd_even_cuts(numtasks, 0);
    for (int i = 0; i < numtasks; ++i) {
        int cur_len = cuts[i + 1] - cuts[i];
        if (cur_len % 2 == 0) odd_even_cuts[i] = 1;
    }
    // printf("taskid: %d || ", taskid);
    // print_vec(vec, cuts[taskid], cuts[taskid+1]-1);
    // int trueEnd = vec.size() - 1; 
    bool sorted = false;
    bool tasksSorted[numtasks];

    int cnt = 0;
    int phase_flag = 0;
    // MPI_Request request;
    // MPI_Status status;
    bool partSorted = false;
    while (!sorted) {
        sorted = true;
        partSorted = true;
        if (numtasks > 1) 
            tPhase(vec, cuts[taskid], cuts[taskid+1] - 1, taskid, numtasks, odd_even_cuts, partSorted);
        ntPhase(vec, cuts[taskid], cuts[taskid+1] - 1, taskid, numtasks, odd_even_cuts, partSorted);
        
        // stop condition
        MPI_Gather(&partSorted, 1, MPI_CXX_BOOL, tasksSorted, 1, MPI_CXX_BOOL, MASTER, MPI_COMM_WORLD);
        for (int i = 0; i < numtasks; ++i) {
            if (!tasksSorted[i]) sorted = false;
        }
        MPI_Bcast(&sorted, 1, MPI_CXX_BOOL, MASTER, MPI_COMM_WORLD);

    }
    // printf("all sorted\n");
    MPI_Barrier(MPI_COMM_WORLD); // sync
    // int* recv_t = vec.data();
    if (taskid == MASTER) {
        // recv_t += cuts[MASTER + 1] - cuts[MASTER];
        for (int t_id  = 1; t_id < numtasks; ++t_id) {
            int recvLength = cuts[t_id + 1] - cuts[t_id];
            // int* recvBuf = (int*)malloc(sizeof(int) * recvLength);
            MPI_Recv(vec.data() + cuts[t_id], recvLength, MPI_INT, t_id, TAG_GATHER, MPI_COMM_WORLD, status);
            // recv_t += recvLength;
            // memcpy(vec.data() + cuts[t_id], recvBuf, sizeof(int) * recvLength);
            // printf("taskid %d received %d\n", t_id, cuts[t_id + 1] - cuts[t_id]);
        }
    } else {
        MPI_Send(vec.data() + cuts[taskid], cuts[taskid + 1] - cuts[taskid], MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // printf("taskid %d sent %d\n", taskid, cuts[taskid + 1] - cuts[taskid]);
    }    
    // MPI_Barrier(MPI_COMM_WORLD); // sync


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
        // print_vec(vec, 0, vec.size() - 1);
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}