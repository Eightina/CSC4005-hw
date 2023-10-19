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
    long long *d_dst = (long long*)dst; 
    long long *d_src = (long long*)src; 
    block_size_col /= 2;
    for (int i = 0; i < block_size_col; ++i) {
        d_dst[i] = d_src[i];
    }
}

// void inline no_sse_memcpy32(int *__restrict dst, const int *src, int block_size) {
//     // #pragma GCC unroll 64
//     for (int i = 0; i < block_size; ++i) {
//         dst[i] = src[i];
//     }
// }

void inline load_block(int *__restrict dst, const Matrix& src, int src_row,
                             int src_col, int block_size_row, int block_size_col) {
    // #pragma GCC unroll 64
    for (int i = 0; i < block_size_row; ++i) {
        // memcpy(dst, src[src_row]+src_col, block_size);
        no_sse_memcpy(dst, src[src_row]+src_col, block_size_col);
        dst += block_size_col;
        src_row++;
    }
}

// void inline preload_block32(int *__restrict dst, const Matrix& src, int src_row, int src_col, int block_size) {
//     // #pragma GCC unroll 64
//     for (int i = 0; i < block_size; ++i) {
//         // memcpy(dst, src[src_row]+src_col, block_size);
//         no_sse_memcpy32(dst, src[src_row]+src_col, block_size);
//         dst += block_size;
//         src_row++;
//     }
// }

// void inline kij_mm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
//     for (int k = 0; k < K; ++k) {
//         for (int i = 0; i < M; ++i) {
//             int r = matrix1[i][k];
//             for (int j = 0; j < N; ++j) {
//                 result[i][j] += r * matrix2[k][j];  
//             }
//         }
//     }
// }

// void inline ijk_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
//     for (int i = 0; i < M; i+=block_size) {
//         for (int j = 0; j < N; j+=block_size) {
//             for (int k = 0; k < K; k+=block_size) {
                
//                 for (int i1 = i; i1 < i+block_size; ++i1) {
//                     for (int j1 = j; j1 < j+block_size; ++j1) {
//                         for (int k1= k; k1 < k+block_size; ++k1) {
//                             result[i1][j1] += matrix1[i1][k1] * matrix2[k1][j1];
//                         }
//                     }
//                 }

//             }
//         }
//     }
// }

// void inline kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
//     for (int k = 0; k < K; k+=block_size) {
//         for (int i = 0; i < M; i+=block_size) {
//             // int r = matrix1[i][k];
//             for (int j = 0; j < N; j+=block_size) {

//                 for (int k1 = k; k1 < k+block_size; ++k1) {
//                     for (int i1 = i; i1 < i+block_size; ++i1) {
//                         int r1 = matrix1[i1][k1];
//                         for (int j1 = j; j1 < j+block_size; ++j1) {
//                             result[i1][j1] += r1 * matrix2[k1][j1];  
//                         }
//                     }
//                 }

//             }
//         }
//     }
// }


void inline ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    int block_size_i = assign_block_size(M);
    int block_size_k = assign_block_size(K);
    int block_size_j = assign_block_size(N);
    // const int block_num_i = M / block_size_i;
    // const int block_num_k = K / block_size_k;
    // const int block_num_j = N / block_size_j;
    const int i_res = M % block_size_i;
    const int k_res = K % block_size_k;
    const int j_res = N % block_size_j;
    const int block_range_i = M - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;

    int zeroload_matrix1[block_size_i * block_size_k];
    int zeroload_matrix2[block_size_k * block_size_j];


    for (int i = 0; i < M; i+=block_size_i) {
        if (i == block_range_i) block_size_i = i_res;
        for (int j = 0; j < N; j+=block_size_j) {
            if (j == block_range_j) block_size_j = j_res;
            int kernel_result[block_size_i * block_size_j] = {};
            for (int k = 0; k < K; k+=block_size_k) {
                if (k == block_range_k) block_size_k = k_res;

                //------------------kernel----------------------------

                load_block(zeroload_matrix1, matrix1, i, k, block_size_i, block_size_k);
                load_block(zeroload_matrix2, matrix2, k, j, block_size_k, block_size_j);
                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;  
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        // const int temp_iloc = (i1 - i) * block_size;
                        const int r1_iter_loc = (i1 - i) * block_size_k + (k1 - k);
                        register int r1 = *(zeroload_matrix1 + r1_iter_loc);

                        const int result_iter_loc = (i1 - i) * block_size_j;
                        for (int j1 = j; j1 < j + block_size_j; ++j1) {

                            kernel_result[result_iter_loc + j1 - j] += r1 * zeroload_matrix2[temp_kloc + j1 - j];  

                        }

                    }
                }
                //------------------kernel----------------------------
            }
            for (int row = 0; row < block_size_i; ++row) {
                no_sse_memcpy(result[i + row] + j, &kernel_result[row * block_size_j], block_size_j);
            }
        }
    }




}

// void inline ikj_ikj_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
//     for (int i = 0; i < M; i+=block_size) {
//         for (int k = 0; k < K; k+=block_size) {
//             for (int j = 0; j < N; j+=block_size) {
//                 int kernel_result[block_size * block_size] = {};

//                 int zeroload_matrix1[block_size * block_size];
//                 load_block(zeroload_matrix1, matrix1, i, k, block_size);

//                 //------------------kernel----------------------------
//                 int preload_matrix2[block_size];
//                 // int preload_result[block_size];
//                 // #pragma GCC unroll 64
//                 for (int i1 = i; i1 < i+block_size; ++i1) {

//                     for (int k1 = k; k1 < k+block_size; ++k1) {

//                         no_sse_memcpy(preload_matrix2, matrix2[k1], block_size);

//                         const int r1_iter_loc = (i1 - i) * block_size + (k1 - k);
//                         register int r1 = *(zeroload_matrix1 + r1_iter_loc);
//                         const int temp_loc = (i1 - i) * block_size;
//                         // #pragma GCC unroll 16
//                         for (int j1 = j; j1 < j + block_size; ++j1) {
//                             kernel_result[temp_loc + j1-j] += r1 * preload_matrix2[j1-j];  
//                         }

//                     }
//                 }
//                 //------------------kernel----------------------------
//                 for (int row = 0; row < block_size; ++row) {
//                     no_sse_memcpy_add(result[i + row] + j, &kernel_result[row * block_size], block_size);
//                 }
//             }


//         }
//     }
// }


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


    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // Considering Memory Locality and Avoiding Cache Missing
    // Hints:
    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication



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