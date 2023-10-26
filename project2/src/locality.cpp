#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <string.h>

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

void inline no_sse_memcpy(void* __restrict dst, const void* __restrict src, int block_size_col) {
    // #pragma GCC unroll 64
    bool odd = block_size_col % 2;
    long long *d_dst = (long long*)dst; 
    long long *d_src = (long long*)src; 
    block_size_col /= 2;
    int i;
    for (i = 0; i < block_size_col; ++i) {
        d_dst[i] = d_src[i];
    }
    if (odd) {
        ((int*)dst)[2*block_size_col] = ((int*)src)[2*block_size_col];
    }
}

void inline load_block(int *__restrict dst, const Matrix& src, int src_row,
                             int src_col, int block_size_row, int block_size_col) {
    // #pragma GCC unroll 64
    for (int i = 0; i < block_size_row; ++i) {
        no_sse_memcpy(dst, src[src_row]+src_col, block_size_col);
        dst += block_size_col;
        src_row++;
    }
}

void inline ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    printf("M:%d, N:%d, K:%d\n", M, N, K);
    const int std_block_size_i = assign_block_size(M);
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(N);
    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;
    printf("blk_M:%d, blk_N:%d, blk_K:%d\n", block_size_i, block_size_j, block_size_k);

    const int i_res = M % block_size_i;
    const int k_res = K % block_size_k;
    const int j_res = N % block_size_j;
    const int block_range_i = M - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;

    int* zeroload_matrix1 = (int*)aligned_alloc(64, (block_size_i * block_size_k + 8) * sizeof(int));
    int* zeroload_matrix2 = (int*)aligned_alloc(64, (block_size_k * block_size_j + 8) * sizeof(int));
    int* kernel_result = (int*)aligned_alloc(64, (block_size_i * block_size_j + 8) * sizeof(int));

    for (int i = 0; i <= block_range_i;) {
        if (i == M) break;
        if (i == block_range_i) {
            block_size_i = i_res;
            i_switch = true;
        }
        for (int j = 0; j <= block_range_j;) {
            if (j == N) break;
            if (j == block_range_j) {
                block_size_j = j_res;
                j_switch = true;
            }
            // int kernel_result[block_size_i * block_size_j] = {};
            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(int));
            for (int k = 0; k <= block_range_k;) {
                if (k == K) break;
                if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }
                //------------------kernel----------------------------

                load_block(zeroload_matrix1, matrix1, i, k, block_size_i, block_size_k);
                load_block(zeroload_matrix2, matrix2, k, j, block_size_k, block_size_j);

                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;  
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        const int r1_iter_loc = (i1 - i) * block_size_k + (k1 - k);
                        register int r1 = *(zeroload_matrix1 + r1_iter_loc);

                        const int result_iter_loc = (i1 - i) * block_size_j;
                        for (int j1 = j; j1 < j + block_size_j; ++j1) {

                            kernel_result[result_iter_loc + j1 - j] += r1 * zeroload_matrix2[temp_kloc + j1 - j];  

                        }

                    }
                }
                //------------------kernel----------------------------
                k += block_size_k;
                if (k_switch) {
                    block_size_k = std_block_size_k;
                    k_switch = false;
                } 
            }
            for (int row = 0; row < block_size_i; ++row) {
                no_sse_memcpy(result[i + row] + j, &kernel_result[row * block_size_j], block_size_j);
            }
            j += block_size_j;
            if (j_switch) {
                block_size_j = std_block_size_j;
                j_switch = false;
            }
        }
        i += block_size_i;
        if (i_switch) {
            block_size_i = std_block_size_i;
            i_switch = false;
        }
    }
    free(zeroload_matrix1);
    free(zeroload_matrix2);
    free(kernel_result);
}

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // kij_mm(M, N, K, matrix1, matrix2, result);       // kij 1024*1024:3816ms; 

    // ijk_tmm(M, N, K, matrix1, matrix2, result, 8);       // ijk 1024*1024:6132ms; 
    // ijk_tmm(M, N, K, matrix1, matrix2, result, 16);      // ijk 1024*1024:6447ms; 
    // ijk_tmm(M, N, K, matrix1, matrix2, result, 32);      // ijk 1024*1024:6202ms; 
    // ijk_tmm(M, N, K, matrix1, matrix2, result, 64);      // ijk 1024*1024:6116ms; best block_size = 64
    // ijk_tmm(M, N, K, matrix1, matrix2, result, 128);     // ijk 1024*1024:6195ms; 
    // ijk_tmm(M, N, K, matrix1, matrix2, result, 512);     // ijk 1024*1024:7466ms;

    // kij_tmm(M, N, K, matrix1, matrix2, result, 16);      // kij 1024*1024:4688ms; 
    // kij_tmm(M, N, K, matrix1, matrix2, result, 64);      // kij 1024*1024:4543ms; 
    // kij_tmm(M, N, K, matrix1, matrix2, result, 4);       // kij 1024*1024:5275ms; 

    // ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 32);     // ijk_kij 1024:3693ms; 

    ijk_kij_tmm(M, N, K, matrix1, matrix2, result);     
    // ijk_kij 1024:3674ms; preload1024:1983ms; betterpreload:1652ms; load_res_block:829ms; 
    // load_res&mat1_block:790ms; +eliminate_useless_inits:774ms; 64_load+mem_align:769ms

    // ijk_ijk_tmm(M, N, K, matrix1, matrix2, result, 64);    // ijk_kij 1024:4571ms; 
    
    // ijk_ikj_tmm(M, N, K, matrix1, matrix2, result, 64);    // ijk_kij 1024:2256; 

    // ikj_ikj_tmm(M, N, K, matrix1, matrix2, result, 64);

    return result;
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

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

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