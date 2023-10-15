//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <string.h>

void inline preload_block(int *__restrict dst, const Matrix& src, int src_row, int src_col, int block_size) {
    #pragma GCC unroll 64
    for (int i = 0; i < block_size; ++i) {
        memcpy(dst, src[src_row]+src_col, block_size);
        dst += block_size;
        src_row++;
    }
}

void inline kij_mm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result) {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            int r = matrix1[i][k];
            for (int j = 0; j < N; ++j) {
                result[i][j] += r * matrix2[k][j];  
            }
        }
    }
}

void inline ijk_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
    for (int i = 0; i < M; i+=block_size) {
        for (int j = 0; j < N; j+=block_size) {
            for (int k = 0; k < K; k+=block_size) {
                
                for (int i1 = i; i1 < i+block_size; ++i1) {
                    for (int j1 = j; j1 < j+block_size; ++j1) {
                        for (int k1= k; k1 < k+block_size; ++k1) {
                            result[i1][j1] += matrix1[i1][k1] * matrix2[k1][j1];
                        }
                    }
                }

            }
        }
    }
}

void inline kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
    for (int k = 0; k < K; k+=block_size) {
        for (int i = 0; i < M; i+=block_size) {
            // int r = matrix1[i][k];
            for (int j = 0; j < N; j+=block_size) {

                for (int k1 = k; k1 < k+block_size; ++k1) {
                    for (int i1 = i; i1 < i+block_size; ++i1) {
                        int r1 = matrix1[i1][k1];
                        for (int j1 = j; j1 < j+block_size; ++j1) {
                            result[i1][j1] += r1 * matrix2[k1][j1];  
                        }
                    }
                }

            }
        }
    }
}


void inline ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
    for (int i = 0; i < M; i+=block_size) {
        for (int j = 0; j < N; j+=block_size) {
            // int zeroload_result[block_size * block_size];
            // preload_block(zeroload_result, result[i]+j, block_size);
            for (int k = 0; k < K; k+=block_size) {
                
                // register int zeroload_matrix1[block_size * block_size];
                // preload_block(zeroload_matrix1, matrix1, i, k, block_size);
                // register int zeroload_matrix2[block_size * block_size];
                // preload_block(zeroload_matrix2, matrix2, k, j, block_size);

                //------------------kernel----------------------------
                // #pragma GCC unroll 64
                for (int k1 = k; k1 < k+block_size; ++k1) {
                    // #pragma GCC unroll 64
                    int preload_matrix2[block_size];
                    // memcpy(preload_matrix2, zeroload_matrix2 + ((k1 - k)*block_size), block_size);
                    memcpy(preload_matrix2, matrix2[k1], block_size);

                    for (int i1 = i; i1 < i+block_size; ++i1) {
                        
                        // register int r1 = *(zeroload_matrix1+(i1 - i)*block_size+(k1-k));
                        register int r1 = matrix1[i1][k1];

                        int preload_result[block_size];
                        memcpy(preload_result, result[i1], block_size);
                        // #pragma GCC unroll 64
                        for (int jx = 0; jx < block_size; ++jx) {
                            preload_result[jx] += r1 * preload_matrix2[jx];
                        }
                        memcpy(result[i1] + j, preload_result, block_size);  //why is memcpy so fast....
                        // #pragma GCC unroll 64
                        // for (int j1 = j; j1 < j+block_size; ++j1) {
                        //     // result[i1][j1] = preload_result[j1-j] + r1 * preload_matrix2[j1-j];  
                        //     result[i1][j1] = preload_result[j1-j];  
                        // }

                    }
                }
                //------------------kernel----------------------------

            }
        }
    }
}

// void inline kij_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size) {
//     for (int k = 0; k < K; k+=block_size) {
//         for (int i = 0; i < N; i+=block_size) {
//             // int zeroload_result[block_size * block_size];
//             // preload_block(zeroload_result, result[i]+j, block_size);
//             for (int j = 0; j < K; j+=block_size) {
                
//                 // register int zeroload_matrix1[block_size * block_size];
//                 // preload_block(zeroload_matrix1, matrix1, i, k, block_size);
//                 // register int zeroload_matrix2[block_size * block_size];
//                 // preload_block(zeroload_matrix2, matrix2, k, j, block_size);

//                 //------------------kernel----------------------------
//                 // #pragma GCC unroll 64
//                 for (int k1 = k; k1 < k+block_size; ++k1) {
//                     // #pragma GCC unroll 64
//                     register int preload_matrix2[block_size];
//                     // memcpy(preload_matrix2, zeroload_matrix2 + ((k1 - k)*block_size), block_size);
//                     memcpy(preload_matrix2, matrix2[k1], block_size);

//                     for (int i1 = i; i1 < i+block_size; ++i1) {
                        
//                         // register int r1 = *(zeroload_matrix1+(i1 - i)*block_size+(k1-k));
//                         register int r1 = matrix1[i1][k1];

//                         // register int preload_result[block_size];
//                         // memcpy(preload_result, result[i1], block_size);
//                         // #pragma GCC unroll 64
//                         for (int j1 = j; j1 < j+block_size; ++j1) {
//                             // result[i1][j1] = preload_result[j1-j] + r1 * preload_matrix2[j1-j];  
//                             result[i1][j1] += r1 * preload_matrix2[j1-j];  
//                         }

//                     }
//                 }
//                 //------------------kernel----------------------------

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
    ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 64);     // ijk_kij 1024:3674ms; preload1024:1983ms;
    // ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 128);    // ijk_kij 1024:4571ms; 

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