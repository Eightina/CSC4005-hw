//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Merge Sort
//

#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include "../utils.hpp"

int THREAD_NUM = 1;
struct ThreadData {
    std::vector<int>* vec;
    int l;
    int r;
    int threadsLim;
    pthread_mutex_t* mutex;
};

// void merge(int* nums, int l, int m, int r) {
void merge(std::vector<int>& nums, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = nums[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = nums[m + 1 + i];
    }

    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = l; // Initial index of the merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            nums[k] = L[i];
            i++;
        } else {
            nums[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        nums[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        nums[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& nums, int l, int r, int threadsLim, pthread_mutex_t* mutex);

// routine function
void* mergeSortRoutine(void* arg) {
    // pthread_barrier_t barrier;
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    // std::vector<int> vec = *(data->vec);
    // std::vector<int>* vec = *(data->vec);
    if (data->l < data->r) {
        int m = data->l + (data->r - data->l) / 2;

        // Sort first and second halves
        bool threadSpawned = false;
        pthread_t thread;
        ThreadData threadData0;

        pthread_mutex_lock(data->mutex);
        if (THREAD_NUM + 1 <= data->threadsLim) {
            threadSpawned = true;
            ++THREAD_NUM;
        } 
        pthread_mutex_unlock(data->mutex); 

        // right half
        if (threadSpawned) {
            threadData0.vec = data->vec;
            threadData0.l = m + 1;
            threadData0.r = data->r;
            threadData0.threadsLim = data->threadsLim;
            threadData0.mutex = data->mutex;
            pthread_create(&thread, nullptr, mergeSortRoutine, &threadData0);
        } else {
            mergeSort(*(data->vec), m + 1, data->r, data->threadsLim, data->mutex);
        }

        // left half
        mergeSort(*(data->vec), data->l, m, data->threadsLim, data->mutex);

        if (threadSpawned) pthread_join(thread, nullptr);

        // Merge the sorted halves
        // pthread_mutex_lock(data->mutex);
        merge(*(data->vec), data->l, m, data->r);
        // pthread_exit(0);
        // print_vec(*(data->vec), 0, (*(data->vec)).size() - 1);
        // pthread_mutex_unlock(data->mutex); 
        
    }
    // pthread_mutex_lock(data->mutex);
    // --THREAD_NUM;
    // pthread_mutex_unlock(data->mutex); 
    return nullptr;
}


// Main function to perform merge sort on a vector v[]
void mergeSort(std::vector<int>& vec, int l, int r, int threadsLim, pthread_mutex_t* mutex) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        bool threadSpawned = false;
        pthread_t thread;
        ThreadData threadData0;

        pthread_mutex_lock(mutex);
        if (THREAD_NUM + 1 <= threadsLim) {
            threadSpawned = true;
            ++THREAD_NUM;
        } 
        pthread_mutex_unlock(mutex); 

        // right half
        if (threadSpawned) {
            threadData0.vec = &vec;
            threadData0.l = m + 1;
            threadData0.r = r;
            threadData0.threadsLim = threadsLim;
            threadData0.mutex = mutex;
            pthread_create(&thread, nullptr, mergeSortRoutine, &threadData0);
        } else {
            mergeSort(vec, m + 1, r, threadsLim, mutex);
        }

        // left half
        mergeSort(vec, l, m, threadsLim, mutex);

        if (threadSpawned) pthread_join(thread, nullptr);

        // Merge the sorted halves
        merge(vec, l, m, r);
        // print_vec(vec, 0, vec.size() - 1);
        // pthread_mutex_unlock(mutex); 
    }   
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

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    // print_vec(vec, 0, vec.size() - 1);
    mergeSort(vec, 0, size - 1, thread_num, &mutex);
    // print_vec(vec, 0, vec.size() - 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}