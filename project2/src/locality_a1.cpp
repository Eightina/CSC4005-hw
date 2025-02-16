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
            int kernel_result[block_size_i * block_size_j] = {};
            for (int k = 0; k <= block_range_k;) {
                if (k == K) break;
                if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }
                //------------------kernel----------------------------

                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;  
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        register int r1 = matrix1[i1][k1];

                        const int result_iter_loc = (i1 - i) * block_size_j;
                        for (int j1 = j; j1 < j + block_size_j; ++j1) {
                            result[i1][j1] += r1 * matrix2[k1][j1];  
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

}


Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    ijk_kij_tmm(M, N, K, matrix1, matrix2, result);     

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