#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cuda_runtime.h> // CUDA Header
#include <string.h>

__global__ void cuda_kernel(int M, int N, int K, int* matrix1, int* matrix2, int* result) {
}


void matrix_multiply_cuda(const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();


    // ijk_kij_tmm(M, N, K, matrix1, matrix2, result);     
    // run the kernel

    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuDuration, start, stop);

    // return result;
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

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    Matrix result(matrix1.getRows(), matrix2.getCols());


    // Matrix result = matrix_multiply_cuda(matrix1, matrix2);


    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}