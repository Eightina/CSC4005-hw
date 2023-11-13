//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort
//

#include <iostream>
#include <vector>
#include <queue>
#include "string.h"
#include "../utils.hpp"

int THREAD_NUM = 1;
// int minRange = 100000000 / 10;
int minRange = 1;

struct ThreadData {
    std::vector<int>* vec;
    int low;
    int high;
    int threadsLim;
    pthread_mutex_t* mutex;  
};

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

void* quickSortRoutine(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    if (data->low < data->high) {
        int pivotIndex = partition(*(data->vec), data->low, data->high);
        bool threadSpawned = false;

        pthread_mutex_lock(data->mutex);
        if (THREAD_NUM + 1 <= data->threadsLim && ((data->high - pivotIndex - 1 + 1) >= minRange)) {
        // if (THREAD_NUM + 1 <= data->threadsLim) {
            threadSpawned = true;
            ++THREAD_NUM;
        } 
        pthread_mutex_unlock(data->mutex); 

        pthread_t thread;
        ThreadData newThreadData0, newThreadData1;

        // right half
        newThreadData0.vec = data->vec;
        newThreadData0.low = pivotIndex + 1;
        newThreadData0.high = data->high;
        newThreadData0.threadsLim = data->threadsLim;
        newThreadData0.mutex = data->mutex;
        if (threadSpawned) {
            // printf("new %d thread spliting %d ~ %d, total: %d\n", 1, pivotIndex + 1, data->high, THREAD_NUM);
            pthread_create(&thread, nullptr, quickSortRoutine, &newThreadData0);
        } else {
            quickSortRoutine(&newThreadData0);
        }

        // left half
        newThreadData1.vec = data->vec;
        newThreadData1.low = data->low;
        newThreadData1.high = pivotIndex - 1;
        newThreadData1.threadsLim = data->threadsLim;
        newThreadData1.mutex = data->mutex;
        quickSortRoutine(&newThreadData1);

        if (threadSpawned) {
            pthread_join(thread, nullptr);
            // printf("1 thread recycled, total: %d\n", THREAD_NUM - 1);
            --THREAD_NUM;
        }
        // quickSortRoutine(*(data->vec), data->low, pivotIndex - 1);
        // quickSortRoutine(*(data->vec), pivotIndex + 1, data->high);
    }
}

void quickSort(std::vector<int>& vec, int threadsLim, int low, int high) {
    // int *nums = vec.data();
    // std::vector<int>* res = new std::vector<int>(cuts[numtasks], 0);
    ThreadData data;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    data.vec = &vec;
    data.low = low;
    data.high = high;
    data.threadsLim = threadsLim;
    data.mutex = &mutex;

    quickSortRoutine(&data);

}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
            );
    }

    const int thread_num = atoi(argv[1]);

    const int size = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, thread_num, 0, size - 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}