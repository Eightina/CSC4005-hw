#include <immintrin.h>
#include <omp.h> 
#include <string.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

void inline avx_memcpy(void* __restrict dst, const void* __restrict src, int block_size) {
    __m256i *d_dst = (__m256i*)dst;
    __m256i *d_src = (__m256i*)src; 
    block_size /= 8;
    for (int i = 0; i < block_size; ++i) {
        d_dst[i] = d_src[i];
    }
}

void inline preload_block(int *__restrict dst, const Matrix& src, int src_row, int src_col, int block_size) {
    // #pragma GCC unroll 64
    for (int i = 0; i < block_size; ++i) {
        // memcpy(dst, src[src_row]+src_col, block_size);
        avx_memcpy(dst, src[src_row]+src_col, block_size);
        dst += block_size;
        src_row++;
    }
}

void inline simd_ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
#pragma omp parallel shared(M, N, K, matrix1, matrix2, result, block_size) 
{
    int* zeroload_matrix1 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
    int* zeroload_matrix2 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));

    #pragma omp for
    for (int i = 0; i < M; i+=block_size) {
        // printf("Hey, I'm thread %d inside the par zone!\n", omp_get_thread_num()); 
        for (int j = 0; j < N; j+=block_size) {
            // int kernel_result[block_size * (block_size + 8)] = {};
            int* kernel_result = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
            memset(kernel_result, 0, block_size * block_size * sizeof(int));

            for (int k = 0; k < K; k+=block_size) {

                //------------------kernel----------------------------
                preload_block(zeroload_matrix1, matrix1, i, k, block_size);
                preload_block(zeroload_matrix2, matrix2, k, j, block_size);

                for (int k1 = k; k1 < k+block_size; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size;  

                    for (int i1 = i; i1 < i+block_size; ++i1) {
                        const int temp_iloc = (i1 - i) * block_size;
                        const int r1_iter_loc = temp_iloc + (k1 - k);
                        register __m512i r1 = _mm512_set1_epi32(*(zeroload_matrix1 + r1_iter_loc));

                        for (int j1 = j; j1 < j + block_size; j1 += 16) {
                            __m512i kernel_res_512 = _mm512_load_epi32(kernel_result + temp_iloc + j1 - j);  
                            __m512i matrix2_512 = _mm512_load_epi32(zeroload_matrix2 + temp_kloc + j1 - j);
                            __m512i mul_res = _mm512_mullo_epi32(r1, matrix2_512); // dont use _mul_epi :(
                            kernel_res_512 = _mm512_add_epi32(kernel_res_512, mul_res);
                            _mm512_store_epi32(kernel_result + temp_iloc + j1 - j, kernel_res_512);
                        }
                    }
                }
                //------------------kernel----------------------------

            }

            for (int row = 0; row < block_size; ++row) {
                avx_memcpy(result[i + row] + j, &kernel_result[row * block_size], block_size);
            }

        }
    }
}
}

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2, int thread_num) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 64); 
    // // Your Code Here!
    // // Optimizing Matrix Multiplication 
    // // In addition to SIMD, Memory Locality and Cache Missing,
    // // Further Applying OpenMp

    
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}