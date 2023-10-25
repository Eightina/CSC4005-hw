#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cuda_runtime.h> // CUDA Header
#include <string.h>

inline void cudaCheckError() {
    cudaError_t error_0 = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error_0));
}

__global__ void cudaKernel(int M, int N, int K, int* matrix1, int* matrix2, int* result) {
    int tn = blockIdx.x * blockDim.x + threadIdx.x;
    int tm = blockIdx.y * blockDim.y + threadIdx.y;
    if (tm < M && tn < N) {
        int temp = 0;
        for (int k = 0; k < K; ++k) {
            temp += matrix1[tm * K + k] * matrix2[k * N + tn];
            // printf("(%d, %d) += %d * %d \n", tm, tn, matrix1[tm * K + k], matrix2[k * N + tn]);
        }
        result[tm * N + tn] += temp;
    }
}

// void seematrx(int* d_matrix1, int M, int K) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < K; ++i) {
//             printf("%d ", d_matrix1[i*K + j]);
//         }
//         printf("\n");
//     }
// }

float matrix_multiply_cuda(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    // gpu mem allocation
    int *d_matrix1, *d_matrix2, *d_result;
    cudaMalloc((void**)&d_matrix1, M * K * sizeof(int));
    cudaMalloc((void**)&d_matrix2, K * N * sizeof(int));
    cudaMalloc((void**)&d_result, M * N * sizeof(int));

    // check and assign heap memory
    // size_t cuda_heap_size = 0;
    // cudaDeviceGetLimit(&cuda_heap_size, cudaLimitMallocHeapSize);
    // printf("before: heap size is %d MB\n", cuda_heap_size / 1024 / 1024);
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, M * N * K *sizeof(unsigned char));
    // cudaDeviceGetLimit(&cuda_heap_size, cudaLimitMallocHeapSize);
    // printf("after: heap size is %d MB\n", cuda_heap_size / 1024 / 1024);

    // copy input data from host to device
    for (int i = 0; i < M; ++i) cudaMemcpy(d_matrix1 + i * K, matrix1[i], K*sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < K; ++i) cudaMemcpy(d_matrix2 + i * N, matrix2[i], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    // seematrx(d_matrix1, M, K);
    // seematrx(d_matrix2, K, N);

    // start time counting
    cudaEvent_t start, stop;
    float gpuDuration = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run the kernel
    int blockdimx = 32;
    int blockdimy = 4;
    const dim3 blockshape(blockdimx, blockdimy);
    const dim3 gridshape((N + blockdimx - 1) / blockdimx, (M + blockdimy - 1) / blockdimy);
    cudaKernel<<<gridshape, blockshape>>>(
        M, N, K, d_matrix1, d_matrix2, d_result
    );
    cudaCheckError();

    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuDuration, start, stop);
    
    for (int i = 0; i < M; ++i) cudaMemcpy(result[i], d_result+ i * N, N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaCheckError();

    return gpuDuration;
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    // cpu mem allocation
    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    Matrix result(matrix1.getRows(), matrix2.getCols());

    float elapsed_time = matrix_multiply_cuda(matrix1, matrix2, result);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time << " milliseconds"
              << std::endl;

    return 0;
}
