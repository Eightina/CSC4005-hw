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
int minRange = 1;
int minPartitionRange = 1;
int minPrefixsumRange = 1;

struct ThreadData {
    std::vector<int>* vec;
    int low;
    int high;
    int threadsLim;
    pthread_mutex_t* mutex;  
};

struct PartitionThreadData {
    std::vector<int>* vec;
    int low;
    int high;
    int threadsLim;
    int pivotIdx;
    pthread_mutex_t* mutex;  
};
struct GenSThreadData {
    int tid;
    std::vector<int>* vec;
    std::vector<int>* S;
    std::vector<int>* L;
    int low;
    int high;
    int pivotVal;
};
struct PrefixThreadData {
    int tid;
    std::vector<int>* S;
    std::vector<int>* L;
    std::vector<int>* wS;
    std::vector<int>* wL;
    int low;
    int high;
    // int pivotVal;
    int threadsLim;
    pthread_mutex_t* mutex;  

};

struct FinalThreadData {
    int tid;
    int pivotVal;
    std::vector<int>* vec;
    std::vector<int>* S;
    std::vector<int>* L;
    int low;
    int high;
    int j;
};

int partition(std::vector<int> &vec, int low, int high) {
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

void* genSRoutine(void* arg) { //generate S, L
    GenSThreadData* data = reinterpret_cast<GenSThreadData*>(arg);
    std::vector<int>* vec = data->vec; 
    std::vector<int>* S = data->S; 
    std::vector<int>* L = data->L; 
    for (int i = data->low; i < data->high; ++i) {
        int temp = (int)((*vec)[i] <= data->pivotVal);
        (*S)[i] = temp;
        (*L)[i] = 1 - temp;
    }
    THREAD_NUM--;
}

// generate wS and wL
void* prefixSumRoutine(void* arg) {
    PrefixThreadData* data = reinterpret_cast<PrefixThreadData*>(arg);
    int low = data->low, high = data->high;
    std::vector<int> *wS = data->wS, *wL = data->wL, *S = data->S, *L = data->L;
    if (high = low + 1) {
        (*wS)[low / 2] = (*S)[low] + (*S)[high]; 
        (*wL)[low / 2] = (*L)[low] + (*L)[high]; 
        return nullptr;
    }

    bool threadSpawned = false;
    pthread_t thread;
    PrefixThreadData newThreadData0, newThreadData1;
    int mid = low + (high - low) / 2;
    
    pthread_mutex_lock(data->mutex);
    if (THREAD_NUM + 1 <= data->threadsLim && ((data->high - data->low) >= minPrefixsumRange)) {
        threadSpawned = true;
        ++THREAD_NUM;
    } 
    pthread_mutex_unlock(data->mutex); 

    // right half
    newThreadData1.tid = 1;
    newThreadData1.S = data->S;
    newThreadData1.L = data->L;
    newThreadData1.wS = data->wS;
    newThreadData1.wL = data->wL;
    newThreadData1.low = mid + 1;
    newThreadData1.high = high;
    newThreadData1.threadsLim = data->threadsLim;
    newThreadData1.mutex = data->mutex;
    if (threadSpawned) {
        pthread_create(&thread, nullptr, prefixSumRoutine, &newThreadData1);
    } else {
        prefixSumRoutine(&newThreadData1);
    }

    // left half
    newThreadData0.tid = 0;
    newThreadData0.S = data->S;
    newThreadData0.L = data->L;
    newThreadData0.wS = data->wS;
    newThreadData0.wL = data->wL;
    newThreadData0.low = low;
    newThreadData0.high = mid;
    newThreadData0.threadsLim = data->threadsLim;
    newThreadData0.mutex = data->mutex;
    prefixSumRoutine(&newThreadData0);

    if (threadSpawned) pthread_join(thread, nullptr);

    int len = high - low + 1;
    (*wS)[high / 2] = (*wS)[high / 2] + (*wS)[high / 2 - len / 4]; 
    (*wL)[high / 2] = (*wL)[high / 2] + (*wL)[high / 2 - len / 4]; 
    
    --THREAD_NUM;
}

void* addxRoutine(void* arg) {
    PrefixThreadData* data = reinterpret_cast<PrefixThreadData*>(arg);
    std::vector<int>* S = data->S; 
    std::vector<int>* L = data->L; 
    std::vector<int>* wS = data->wS; 
    std::vector<int>* wL = data->wL; 
    if (data->low == 0) {
        data->low = 1;
    }
    for (int i = data->low; i <= data->high; ++i) {
        (*S)[i] = ((i / 2) == 0) ? ((*wS)[i / 2] + (*S)[i]) : ((*wS)[i / 2]);
        (*L)[i] = ((i / 2) == 0) ? ((*wL)[i / 2] + (*L)[i]) : ((*wL)[i / 2]);
    }
    --THREAD_NUM;
}

void* finalRoutine(void* arg) {
    FinalThreadData* data = reinterpret_cast<FinalThreadData*>(arg);
    std::vector<int>* vec = data->vec; 
    std::vector<int>* S = data->S; 
    std::vector<int>* L = data->L; 
    for (int i = data->low; i <= data->high; ++i) {
        int x = (*vec)[i];
        if (x < data->pivotVal) (*vec)[(*S)[i] - 1] = x;
        else (*vec)[data->j + (*L)[i]] = x;
    }
    (*vec)[data->j] = data->pivotVal;
    --THREAD_NUM;
}

int parallelPartition(void* arg) {
    PartitionThreadData* data = reinterpret_cast<PartitionThreadData*>(arg);
    std::vector<int>* vec = data->vec; 
    int piovtVal = (*vec)[data->pivotIdx];

    if (data->high - data->low < minPartitionRange) {
        int res = partition(*vec, data->low, data->high);
        --THREAD_NUM;
        return res;
    }

    pthread_mutex_lock(data->mutex);
    int encodeThreadsNum = (data->threadsLim - THREAD_NUM) * (data->high - data->low + 1) / vec->size();
    if (encodeThreadsNum < 1) {
        pthread_mutex_unlock(data->mutex);
        int res = partition(*vec, data->low, data->high);
        --THREAD_NUM;
        return res;
    } 
    THREAD_NUM += encodeThreadsNum;
    printf("%d + 1 threads encoding %d ~ %d\n", encodeThreadsNum, data->low, data->high);
    pthread_mutex_unlock(data->mutex);

    // generate S, L array in parallel
    std::swap((*vec)[data->high], (*vec)[data->pivotIdx]);
    int SLen = (data->high) - (data->low) + 1 - 1;
    int subLen = SLen / (encodeThreadsNum + 1);
    pthread_t threads[encodeThreadsNum + 1];
    GenSThreadData datas[encodeThreadsNum + 1];
    std::vector<int> S(SLen, 0);
    std::vector<int> L(SLen, 0);
    for (int tid = 0; tid < encodeThreadsNum + 1; ++tid) {
        datas[tid].tid = tid;
        datas[tid].vec = vec;
        datas[tid].low = tid * subLen;
        datas[tid].high = (tid + 1) * subLen;
        datas[tid].S = &S;
        datas[tid].L = &L;
        datas[tid].pivotVal = piovtVal;
        if (tid != encodeThreadsNum) {
            pthread_create(&threads[tid], nullptr, genSRoutine, &datas[tid]);
            continue;
        }
        datas[tid].high = data->high - 1;
        genSRoutine(&datas[tid]);
    }
    for (int tid = 0; tid < encodeThreadsNum; ++tid) {
        pthread_join(threads[tid], nullptr);
    }
    // get wS, wL
    std::vector<int> wS(SLen / 2, 0);
    std::vector<int> wL(SLen / 2, 0);
    PrefixThreadData ptData;
    ptData.tid = 0;
    ptData.S = &S;
    ptData.L = &L;
    ptData.wS = &wS;
    ptData.wL = &wL;
    ptData.low = data->low;
    ptData.high = data->high;
    ptData.threadsLim = data->threadsLim;
    ptData.mutex = data->mutex;
    prefixSumRoutine(&ptData);

    // calculate final prefix sum
    pthread_mutex_lock(data->mutex);
    int addxThreadsNum = (data->threadsLim - THREAD_NUM) * (data->high - data->low + 1) / vec->size();
    if (addxThreadsNum < 1) {
        pthread_mutex_unlock(data->mutex);
        for (int i = 1; i < SLen; ++i) {
            S[i] = ((i / 2)== 0) ? (wS[i / 2] + S[i]) : (wS[i / 2]);
            L[i] = ((i / 2)== 0) ? (wL[i / 2] + L[i]) : (wL[i / 2]);
        }
    } 
    THREAD_NUM += addxThreadsNum;
    pthread_mutex_unlock(data->mutex);
    subLen = SLen / (addxThreadsNum + 1);
    pthread_t addxThreads[addxThreadsNum + 1];
    PrefixThreadData addxDatas[addxThreadsNum + 1];
    for (int tid = 0; tid < addxThreadsNum + 1; ++tid) {
        addxDatas[tid].tid = tid;
        addxDatas[tid].low = tid * subLen;
        addxDatas[tid].high = (tid + 1) * subLen;
        addxDatas[tid].S = &S;
        addxDatas[tid].wS = &wS;
        addxDatas[tid].L = &L;
        addxDatas[tid].wL = &wL;
        if (tid != addxThreadsNum) {
            pthread_create(&addxThreads[tid], nullptr, addxRoutine, &addxDatas[tid]);
            continue;
        }
        addxDatas[tid].high = data->high - 1;
        genSRoutine(&addxDatas[tid]);
    }
    for (int tid = 0; tid < addxThreadsNum; ++tid) {
        pthread_join(addxThreads[tid], nullptr);
    }

    // final step of partitioning
    int j = S[SLen - 1] + 1;
    pthread_mutex_lock(data->mutex);
    int finalThreadsNum = (data->threadsLim - THREAD_NUM) * (data->high - data->low + 1) / vec->size();
    if (finalThreadsNum < 1) {
        pthread_mutex_unlock(data->mutex);
        for (int i = 0; i < SLen - 1; ++i) {
            int x = (*vec)[i];
            if (x < piovtVal) (*vec)[S[i] - 1] = x;
            else (*vec)[j + L[i]] = x;
        }
        (*vec)[j] = piovtVal;
    } 
    THREAD_NUM += finalThreadsNum;
    pthread_mutex_unlock(data->mutex);
    subLen = SLen / (finalThreadsNum + 1);
    pthread_t finalThreads[finalThreadsNum + 1];
    FinalThreadData finalDatas[finalThreadsNum + 1];
    for (int tid = 0; tid < finalThreadsNum + 1; ++tid) {
        finalDatas[tid].tid = tid;
        finalDatas[tid].vec = vec;
        finalDatas[tid].low = tid * subLen;
        finalDatas[tid].high = (tid + 1) * subLen;
        finalDatas[tid].S = &S;
        finalDatas[tid].L = &L;
        finalDatas[tid].pivotVal = piovtVal;
        finalDatas[tid].j = j;
        if (tid != encodeThreadsNum) {
            pthread_create(&threads[tid], nullptr, finalRoutine, &finalDatas[tid]);
            continue;
        }
        datas[tid].high = data->high - 1;
        genSRoutine(&datas[tid]);
    }
    for (int tid = 0; tid < finalThreadsNum; ++tid) {
        pthread_join(threads[tid], nullptr);
    }
    
    --THREAD_NUM;
    return j;
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

    // minRange = size / 128;
    minRange = size / 16;
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