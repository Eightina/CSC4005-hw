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
    int* nums;
    int l;
    int r;
    int threadsLim;
    pthread_mutex_t* mutex;
};

void merge(int* nums, int l, int m, int r) {
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

void mergeSort(int* nums, int l, int r, int threadsLim, pthread_mutex_t* mutex);

// routine function
void* mergeSortRoutine(void* arg) {
    
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    if (data->l < data->r) {
        int m = data->l + (data->r - data->l) / 2;

        // Sort first and second halves
        pthread_mutex_lock(data->mutex);
        if (THREAD_NUM + 1 <= data->threadsLim){
            ++THREAD_NUM;
            pthread_mutex_unlock(data->mutex); 
            pthread_t thread0;
            ThreadData threadData0;
            threadData0.nums = data->nums;
            threadData0.l = data->l;
            threadData0.r = m;
            threadData0.threadsLim = data->threadsLim;
            threadData0.mutex = data->mutex;
            pthread_create(&thread0, nullptr, mergeSortRoutine, &threadData0);
        } else {
            pthread_mutex_unlock(data->mutex); 
            mergeSort(data->nums, data->l, m, data->threadsLim, data->mutex);
        }

        pthread_mutex_lock(data->mutex);
        if (THREAD_NUM + 1 <= data->threadsLim) {
            ++THREAD_NUM;
            pthread_mutex_unlock(data->mutex); 

            pthread_t thread0;
            ThreadData threadData0;
            threadData0.nums = data->nums;
            threadData0.l = m + 1;
            threadData0.r = data->r;
            threadData0.threadsLim = data->threadsLim;
            threadData0.mutex = data->mutex;
            pthread_create(&thread0, nullptr, mergeSortRoutine, &threadData0);
        } else {
            pthread_mutex_unlock(data->mutex); 
            mergeSort(data->nums, m + 1, data->r, data->threadsLim, data->mutex);
        }


        // Merge the sorted halves
        merge(data->nums, data->l, m, data->r);
    }

    return nullptr;
    pthread_mutex_lock(data->mutex);
    --THREAD_NUM;
    pthread_mutex_unlock(data->mutex); 
}


// Main function to perform merge sort on a vector v[]
void mergeSort(int* nums, int l, int r, int threadsLim, pthread_mutex_t* mutex) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        pthread_mutex_lock(mutex);
        if (THREAD_NUM + 1 <= threadsLim){
            ++THREAD_NUM;
            pthread_mutex_unlock(mutex); 
            pthread_t thread0;
            ThreadData threadData0;
            threadData0.nums = nums;
            threadData0.l = l;
            threadData0.r = m;
            threadData0.threadsLim = threadsLim;
            pthread_create(&thread0, nullptr, mergeSortRoutine, &threadData0);
        } else {
            pthread_mutex_unlock(mutex); 
            mergeSort(nums, l, m, threadsLim, mutex);
        }

        pthread_mutex_lock(mutex);
        if (THREAD_NUM + 1 <= threadsLim) {
            ++THREAD_NUM;
            pthread_mutex_unlock(mutex); 
            pthread_t thread0;
            ThreadData threadData0;
            threadData0.nums = nums;
            threadData0.l = m + 1;
            threadData0.r = r;
            threadData0.threadsLim = threadsLim;
            pthread_create(&thread0, nullptr, mergeSortRoutine, &threadData0);
        } else {
            pthread_mutex_unlock(mutex); 
            mergeSort(nums, m + 1, r, threadsLim, mutex);
        }


        // Sort first and second halves
        // mergeSort(nums, l, m);
        // mergeSort(nums, m + 1, r);

        // Merge the sorted halves
        merge(nums, l, m, r);
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
    mergeSort(vec.data(), 0, size - 1, thread_num, &mutex);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}