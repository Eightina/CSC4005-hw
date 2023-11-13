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
#include "string.h"
#include "../utils.hpp"

int THREAD_NUM = 1;
// int SPLIT_THREAD_NUM = 1;
// int minRange = 10000;
int minRange = 100;
// int mergeMinRange = 40000000;
int mergeMinRange = 250;

void mergeSort(std::vector<int>& nums, int l, int r, int threadsLim, pthread_mutex_t* mutex);

struct ThreadData {
    std::vector<int>* vec;
    int l;
    int m;
    int r;
    int threadsLim;
    pthread_mutex_t* mutex;
};

struct MergeThreadData {
    int tid;
    std::vector<int>* vec;
    int* res;
    std::vector<int>* kIndexes1;
    std::vector<int>* kIndexes2;
    // int threadsLim;
    pthread_mutex_t* mutex;
};

void* getNKthElements(const std::vector<int>& nums1, const std::vector<int>& nums2, 
                    int numThreads, std::vector<int>& kIndexes1, std::vector<int>& kIndexes2,
                    int offset1, int offset2) {
    int m = nums1.size();
    int n = nums2.size();
    int total = m + n;
    int k = total / numThreads;
    int index1 = 0, index2 = 0;

    int resCount = 0;
    while (resCount != numThreads - 1) {
        // 边界情况
        if (index1 == m) {
            kIndexes1.emplace_back(offset1 + index1 - 1);
            kIndexes2.emplace_back(offset2 + index2 + k - 1);
            return nullptr;
            // return nums2[index2 + k - 1];
        }
        if (index2 == n) {
            kIndexes1.emplace_back(offset1 + index1 + k - 1);
            kIndexes2.emplace_back(offset2 + index2 - 1);
            return nullptr;
            // return nums1[index1 + k - 1];
        }
        if (k == 1) {
            ++resCount;
            if (nums1[index1] <= nums2[index2]) {
                kIndexes1.emplace_back(offset1 + index1);
                kIndexes2.emplace_back(offset2 + index2 - 1);
            } else {
                kIndexes1.emplace_back(offset1 + index1 - 1);
                kIndexes2.emplace_back(offset2 + index2);
            }
            // move on to fine next split
            k = total / numThreads;
            // return std::min(nums1[index1], nums2[index2]);
        }
        // 正常情况
        int newIndex1 = std::min(index1 + k / 2 - 1, m - 1);
        int newIndex2 = std::min(index2 + k / 2 - 1, n - 1);
        int pivot1 = nums1[newIndex1];
        int pivot2 = nums2[newIndex2];
        if (pivot1 <= pivot2) {
            k -= newIndex1 - index1 + 1;
            index1 = newIndex1 + 1;
        }
        else {
            k -= newIndex2 - index2 + 1;
            index2 = newIndex2 + 1;
        }
    }
}

// void merge(int* nums, int l, int m, int r) {
void merge(std::vector<int>& nums, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);
    // Copy data to temporary vectors L[] and R[]
    memcpy(L.data(), nums.data() + l, n1 * sizeof(int));
    memcpy(R.data(), nums.data() + m + 1, n2 * sizeof(int));
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

void* mergeRoutine(void* arg) {
    MergeThreadData* data = reinterpret_cast<MergeThreadData*>(arg);
    std::vector<int>* vec = data->vec;
    int* res = data->res;
    std::vector<int>* kIndexes1 = data->kIndexes1;
    std::vector<int>* kIndexes2 = data->kIndexes2;
    int l = (*kIndexes1)[data->tid] + 1;
    int m1 = (*kIndexes1)[data->tid + 1];
    int m2 = (*kIndexes2)[data->tid] + 1;
    int r = (*kIndexes2)[data->tid + 1];

    int n1 = m1 - l + 1;
    int n2 = r - m2 + 1;
    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);
    // Copy data to temporary vectors L[] and R[]
    memcpy(L.data(), (*vec).data() + l, n1 * sizeof(int));
    memcpy(R.data(), (*vec).data() + m2, n2 * sizeof(int));
    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = 0;
    if (data->tid > 0) {
        k += (*kIndexes1)[data->tid] - (*kIndexes1)[0] + 
             (*kIndexes2)[data->tid] - (*kIndexes2)[0];
    }  // Initial index of the merged subarray
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            res[k] = L[i];
            i++;
        } else {
            res[k] = R[j];
            j++;
        }
        k++;
    }
    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        res[k] = L[i];
        i++;
        k++;
    }
    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        res[k] = R[j];
        j++;
        k++;
    }
    --THREAD_NUM;
    pthread_exit(0);
}

void nThreadsMerge(int threadsLim, std::vector<int>& nums, int l, int m, int r, pthread_mutex_t* mutex) {
    if (r - l < mergeMinRange) {
        merge(nums, l, m, r);
        return;
    }
    std::vector<int> nums1(nums.begin() + l, nums.begin() + m + 1);
    std::vector<int> nums2(nums.begin() + m + 1, nums.begin() + r + 1);
    std::vector<int> kIndexes1, kIndexes2;
    kIndexes1.push_back(l - 1);
    kIndexes2.push_back(m + 1 - 1);

    pthread_mutex_lock(mutex);
    int mergeThreadsNum = (threadsLim - THREAD_NUM);
    if (mergeThreadsNum < 2) {
        pthread_mutex_unlock(mutex);
        merge(nums, l, m, r);
        return;    
    } 
    THREAD_NUM += mergeThreadsNum;
    printf("%d threads merging %d ~ %d\n", mergeThreadsNum, l, r);
    getNKthElements(nums1, nums2, mergeThreadsNum, kIndexes1, kIndexes2, l, m + 1);
    pthread_mutex_unlock(mutex);

    kIndexes1.push_back(l + nums1.size() - 1);
    kIndexes2.push_back(m + 1 + nums2.size() - 1);

    int trueNumThreads = kIndexes1.size() - 1;
    pthread_t threads[trueNumThreads];
    MergeThreadData datas[trueNumThreads];

    // pthread_mutex_lock(mutex);
    // pthread_mutex_unlock(mutex); 
    int* res = (int*)malloc(sizeof(int) * (r - l + 1));
    for (int tid = 0; tid < trueNumThreads; ++tid) {
        datas[tid].tid = tid;
        datas[tid].vec = &nums;
        datas[tid].res = res;
        datas[tid].kIndexes1 = &kIndexes1;
        datas[tid].kIndexes2 = &kIndexes2;
        // datas[tid].m2 = kIndexes2[tid] + 1;
        // datas[tid].r = kIndexes2[tid + 1];
        // datas[tid].threadsLim = data->threadsLim;
        datas[tid].mutex = mutex;
        pthread_create(&threads[tid], nullptr, mergeRoutine, &datas[tid]);
    }

    for (int tid = 0; tid < trueNumThreads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    memcpy(nums.data() + l, res, sizeof(int) * (r - l + 1));

    THREAD_NUM -= mergeThreadsNum;
}

// routine function
void* mergeSortRoutine(void* arg) {
    // pthread_barrier_t barrier;
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    // std::vector<int> vec = *(data->vec);
    // std::vector<int>* vec = *(data->vec);
    int splitThreadsLim = data->threadsLim / 2;
    if (data->l < data->r) {
        int m = data->l + (data->r - data->l) / 2;

        // Sort first and second halves
        bool threadSpawned = false;
        pthread_t thread;
        ThreadData threadData0;

        pthread_mutex_lock(data->mutex);
        if (THREAD_NUM + 1 <= splitThreadsLim && ((data->r - m) >= minRange)) {
        // if (THREAD_NUM + 1 <= data->threadsLim) {
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
            printf("%d threads spliting %d ~ %d\n", 1, m + 1, data->r);
            pthread_create(&thread, nullptr, mergeSortRoutine, &threadData0);
        } else {
            mergeSort(*(data->vec), m + 1, data->r, data->threadsLim, data->mutex);
        }

        // left half
        mergeSort(*(data->vec), data->l, m, data->threadsLim, data->mutex);

        if (threadSpawned) pthread_join(thread, nullptr);

        // Merge the sorted halves
        // pthread_mutex_lock(data->mutex);
        // merge(*(data->vec), data->l, m, data->r);
        nThreadsMerge(data->threadsLim, *(data->vec), data->l, m, data->r, data->mutex);
        // memcpy(data->vec->data(), mergeRes, sizeof(int) * data->vec->size());
        // print_vec(*(data->vec), 0, (*(data->vec)).size() - 1);
        // pthread_mutex_unlock(data->mutex); 
        
    }
    // pthread_mutex_lock(data->mutex);
    --THREAD_NUM;
    pthread_exit(0);
    // pthread_mutex_unlock(data->mutex); 
    // return nullptr;
} 


// Main function to perform merge sort on a vector v[]
void mergeSort(std::vector<int>& vec, int l, int r, int threadsLim, pthread_mutex_t* mutex) {
    if (l < r) {
        int m = l + (r - l) / 2;
        int splitThreadsLim = threadsLim / 2;
        // Sort first and second halves
        bool threadSpawned = false;
        pthread_t thread;
        ThreadData threadData0;

        pthread_mutex_lock(mutex);
        if (THREAD_NUM + 1 <= splitThreadsLim && ((r - m) >= minRange)) {
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
            printf("%d threads spliting %d ~ %d\n", 1, m + 1, r);
            pthread_create(&thread, nullptr, mergeSortRoutine, &threadData0);
        } else {
            mergeSort(vec, m + 1, r, threadsLim, mutex);
        }

        // left half
        mergeSort(vec, l, m, threadsLim, mutex);

        if (threadSpawned) pthread_join(thread, nullptr);

        // Merge the sorted halves
        // merge(vec, l, m, r);
        
        nThreadsMerge(threadsLim, vec, l, m, r, mutex);
        // memcpy(vec.data(), mergeRes, sizeof(int) * vec.size());
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