//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <queue>
#include "string.h"
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0


struct node{
	int value;
	int* array; //数组索引
	int index; 
    int array_len;
	node(int v, int* a, int i, int a_len){
		value = v;
		array = a;
		index = i;
        array_len = a_len;
	}
	bool operator<(node a)const{
		return value < a.value;
	}
	bool operator>(node a)const{
		return value > a.value;
	}
};
void mergeSorted(int** nums, std::vector<int> cuts, std::vector<int>& res){
    // printf("merging sorted...");
	int array_num = cuts.size() - 1;
    int total_len = cuts[array_num];
	std::priority_queue<node, std::vector<node>, std::greater<node>> order;
    // init priority queue
	for (int i = 0; i < array_num; i++){
		order.push(node(nums[i][0], nums[i], 0, cuts[i + 1] - cuts[i]));
	}
	int* cur_array = 0;
    int nxt_index = 0;
	// priority queue is full
    int cnt = 0;
	while (cnt < total_len){
		node tmp = order.top();//获得优先队列中最小值元素
		res[cnt] = tmp.value;//存入目标数组
        ++cnt;
		cur_array = tmp.array;//最小值元素对应数组
		nxt_index = tmp.index + 1;//最小值元素对应数组内下一个的元素
		order.pop();
		if (nxt_index < tmp.array_len) {
            order.push(node(cur_array[nxt_index], cur_array, nxt_index, tmp.array_len));
		}
	}
    // printf("merging done\n");
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

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status, std::vector<int> cuts) {
    int *nums = vec.data();
    std::vector<int>* res = new std::vector<int>(cuts[numtasks], 0);
    
    quickSortKernel(vec, cuts[taskid], cuts[taskid + 1] - 1);
    if (taskid == MASTER) {
        // int *new_res = (int *)malloc(sizeof(int) * (vec.size()));
        // arrays[0] = res
        // int* master_res = (int*)malloc(sizeof(int) * (cuts[MASTER + 1] - cuts[MASTER]));
        // memcpy(master_res, nums, sizeof(int) * (cuts[MASTER + 1] - cuts[MASTER]));

        int **arrays = (int **)malloc(sizeof(int*) * (numtasks));
        int *res = (int*)malloc(sizeof(int) * cuts[numtasks-1]);

        memcpy(res, nums, sizeof(int) * (cuts[MASTER + 1] - cuts[MASTER]));
        arrays[0] = res;

        for (int t_id = 1; t_id < numtasks; ++t_id) {

            // int* cur_res = (int*)malloc(sizeof(int) * (cuts[t_id + 1] - cuts[t_id]));
            // MPI_Recv(&nums[cuts[t_id]], cuts[t_id + 1] - cuts[t_id], MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            MPI_Recv(res + cuts[t_id], cuts[t_id + 1] - cuts[t_id], MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            arrays[t_id] = res + cuts[t_id];
            // printf("taskid %d received %d\n", t_id, cuts[t_id + 1] - cuts[t_id]);
        }
        // printf("all tasks received\n");

        if (numtasks >= 2) {
            // printf("res init\n");
            mergeSorted(arrays, cuts, vec);
        }
        free(res);
        free(arrays);
    } else {
        MPI_Send(&nums[cuts[taskid]], cuts[taskid + 1] - cuts[taskid], MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // printf("taskid %d sent %d\n", taskid, cuts[taskid + 1] - cuts[taskid]);
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
    // print_vec(vec, 0, vec.size() -1);
    std::vector<int> vec_clone = vec;

    // job patition
    std::vector<int> cuts = createCuts(0, vec.size() - 1, numtasks);
    // assign_cuts(size, numtasks, cuts);
    // createCuts(0, )

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