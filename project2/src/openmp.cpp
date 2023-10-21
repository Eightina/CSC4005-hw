#include <immintrin.h>
#include <omp.h> 
#include <string.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

int inline assign_block_size(int M) {   
    if (M >= 64) {
        return 64;
    } else if (M >= 32) {
        return 32;
    } else if (M >= 16) {
        return 16;
    } else if (M >= 8) {
        return 8;
    } else if (M >= 4) {
        return 4;
    } 
    return 1;
}

void inline avx_memcpy(void* __restrict dst, const void* __restrict src, int block_size) {
    memcpy(dst, src, block_size*sizeof(int));
}

void inline preload_block(int *__restrict dst, const Matrix& src, int src_row,
                             int src_col, int block_size_row, int block_size_col) {
    // printf("called range to:%d + %d = %d\n", src_row, block_size_row, src_row + block_size_row);
    for (int i = 0; i < block_size_row; ++i) {
        // if (src_row == 1145) printf(">>??\n");
        avx_memcpy(dst, src[src_row]+src_col, block_size_col);
        dst += block_size_col;
        src_row++;
    }
}

void inline avx256_kernel(int k, int i, int j, int* zeroload_matrix1, int* zeroload_matrix2, int* kernel_result,
                        int block_size_k, int block_size_i, int block_size_j) {

    for (int k1 = k; k1 < k + block_size_k; ++k1) {
        const int temp_kloc = (k1 - k) * block_size_j;  
        for (int i1 = i; i1 < i+block_size_i; ++i1) {
            const int rest = block_size_j % 8;
            const int r1_iter_loc = (i1 - i) * block_size_k + (k1 - k);
            register __m256i r1 = _mm256_set1_epi32(*(zeroload_matrix1 + r1_iter_loc));

            const int result_iter_loc = (i1 - i) * block_size_j;
            int j1;
            for (j1 = j; j1 + 8 <= j + block_size_j; j1 += 8) {
                // kernel_result[temp_loc + j1 - j] += r1 * preload_matrix2[j1-j];
                __m256i kernel_res_256 = _mm256_lddqu_si256((__m256i*)(kernel_result + result_iter_loc + j1 - j));  
                __m256i matrix2_256 = _mm256_lddqu_si256((__m256i*)(zeroload_matrix2 + temp_kloc + j1 - j));
                __m256i mul_res = _mm256_mullo_epi32(r1, matrix2_256); // dont use _mul_epi :(
                kernel_res_256 = _mm256_add_epi32(kernel_res_256, mul_res);
                _mm256_storeu_si256((__m256i*)(kernel_result + result_iter_loc + j1 - j), kernel_res_256);
            }
            if (rest) {
                for (; j1 < j + block_size_j; ++j1) {
                    kernel_result[result_iter_loc + j1 - j] += r1[0] * zeroload_matrix2[temp_kloc + j1 - j];  
                }
            }
            
        }
    }
}

void inline simd_ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int thread_num) {

    // printf("M:%d, N:%d, K:%d\n", M, N, K);
    int row_num_per_thread = M / thread_num;
    int left_row_num = M % thread_num;
    int row_cuts[thread_num + 1];
    row_cuts[0] = 0;
    int divided_left_pixel_num = 0;
    for (int i = 0; i < thread_num; i++) {
        if (divided_left_pixel_num < left_row_num) {
            row_cuts[i+1] = row_cuts[i] + row_num_per_thread + 1;
            divided_left_pixel_num++;
        } else row_cuts[i+1] = row_cuts[i] + row_num_per_thread;
    }


// shared(M, N, K, matrix1, matrix2, result, std_block_size_i, std_block_size_k, std_block_size_j) 
#pragma omp parallel 
{
    int id = omp_get_thread_num();

    const int std_block_size_i = assign_block_size(row_cuts[id+1] - row_cuts[id]);
    if (std_block_size_i == 0) return;
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(N);
    // printf("blk_M:%d, blk_N:%d, blk_K:%d\n", block_size_i, block_size_j, block_size_k);

    const int i_res = (row_cuts[id+1] - row_cuts[id]) % std_block_size_i;
    const int k_res = K % std_block_size_k;
    const int j_res = N % std_block_size_j;
    const int block_range_i = row_cuts[id+1] - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;

    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;

    int* zeroload_matrix1 = (int*)aligned_alloc(64, block_size_i * block_size_k * sizeof(int));
    int* zeroload_matrix2 = (int*)aligned_alloc(64, block_size_k * block_size_j * sizeof(int));
    int* kernel_result = (int*)aligned_alloc(64, block_size_i * block_size_j * sizeof(int));

    for (int i = row_cuts[id]; i < row_cuts[id+1];) {

        for (int j = 0; j < N;) {

            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(int));
            for (int k = 0; k < K;) {

                // if (i == 1145) printf("xxxxx\n");

                // printf("%d load block m1 i:%d k:%d ...\n", id, i, k);
                // printf("caller i:%d\n", i);
                preload_block(zeroload_matrix1, matrix1, i, k, block_size_i, block_size_k);
                // printf("done\n");
                
                // printf("%d load block m2 k:%d j:%d ...\n", id, k, j);
                // printf("caller k:%d\n", k);
                preload_block(zeroload_matrix2, matrix2, k, j, block_size_k, block_size_j);
                // printf("done\n");
                //------------------kernel----------------------------
                avx256_kernel(k, i, j, zeroload_matrix1, zeroload_matrix2, kernel_result,
                                block_size_k, block_size_i, block_size_j);
                //------------------kernel----------------------------
                k += block_size_k;
                if (k_switch) {
                    block_size_k = std_block_size_k;
                    k_switch = false;
                } else if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }

            }
            for (int row = 0; row < block_size_i; ++row) {
                avx_memcpy(result[i + row] + j, &kernel_result[row * block_size_j], block_size_j);
            }
            j += block_size_j;
            if (j_switch) {
                block_size_j = std_block_size_j;
                j_switch = false;
            } else if (j == block_range_j) {
                block_size_j = j_res;
                j_switch = true;
            }

        }
        i += block_size_i;
        if (i_switch) {
            block_size_i = std_block_size_i;
            i_switch = false;
        } else if (i == block_range_i) {
            block_size_i = i_res;
            i_switch = true;
        }

    }
}
}

// void inline simd_ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
// #pragma omp parallel shared(M, N, K, matrix1, matrix2, result, block_size) 
// {
//     int* zeroload_matrix1 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
//     int* zeroload_matrix2 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));

//     #pragma omp for
//     for (int i = 0; i < M; i+=block_size) {
//         // printf("Hey, I'm thread %d inside the par zone!\n", omp_get_thread_num()); 
//         for (int j = 0; j < N; j+=block_size) {
//             // int kernel_result[block_size * (block_size + 8)] = {};
//             int* kernel_result = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
//             memset(kernel_result, 0, block_size * block_size * sizeof(int));

//             for (int k = 0; k < K; k+=block_size) {

//                 //------------------kernel----------------------------
//                 preload_block(zeroload_matrix1, matrix1, i, k, block_size);
//                 preload_block(zeroload_matrix2, matrix2, k, j, block_size);

//                 for (int k1 = k; k1 < k+block_size; ++k1) {
//                     const int temp_kloc = (k1 - k) * block_size;  

//                     for (int i1 = i; i1 < i+block_size; ++i1) {
//                         const int temp_iloc = (i1 - i) * block_size;
//                         const int r1_iter_loc = temp_iloc + (k1 - k);
//                         register __m512i r1 = _mm512_set1_epi32(*(zeroload_matrix1 + r1_iter_loc));

//                         for (int j1 = j; j1 < j + block_size; j1 += 16) {
//                             __m512i kernel_res_512 = _mm512_load_epi32(kernel_result + temp_iloc + j1 - j);  
//                             __m512i matrix2_512 = _mm512_load_epi32(zeroload_matrix2 + temp_kloc + j1 - j);
//                             __m512i mul_res = _mm512_mullo_epi32(r1, matrix2_512); // dont use _mul_epi :(
//                             kernel_res_512 = _mm512_add_epi32(kernel_res_512, mul_res);
//                             _mm512_store_epi32(kernel_result + temp_iloc + j1 - j, kernel_res_512);
//                         }
//                     }
//                 }
//                 //------------------kernel----------------------------

//             }

//             for (int row = 0; row < block_size; ++row) {
//                 avx_memcpy(result[i + row] + j, &kernel_result[row * block_size], block_size);
//             }

//         }
//     }
// }
// }

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2, int thread_num) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result, thread_num); 
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