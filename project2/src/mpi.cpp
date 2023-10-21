//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <string.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0
#define TAG_GATHER 0

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

void assign_cuts(int total_workload, int num_tasks, int* cuts) {
    // int total_pixel_num = input_jpeg.width * input_jpeg.height;
    int pixel_num_per_task = total_workload / num_tasks;
    int left_pixel_num = total_workload % num_tasks;

    // std::vector<int> cuts(num_tasks + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < num_tasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }
}
// int assign_block_size(Matrix& matrix1, Matrix& matrix2) {
//     size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
//     if ()
//     int i = 2, n = 0; 
//     int res = ;

// }
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

void inline store_res(Matrix& dst, const int* src, int loc_row, int loc_col,
                             int block_size_i, int block_size_j) {
    const int row_range = loc_row + block_size_i;
    int src_loc = 0;
    for (int i = loc_row; i < row_range; ++i) {
        // memcpy(dst, src[src_row]+src_col, block_size);
        // dst[i] 
        memcpy(dst[i] + loc_col, &src[src_loc], block_size_j*(sizeof(int)));
        src_loc += block_size_j;
        // dst += block_size_;
        // src_row++;
    }
}

void inline load_res(int* dst, const Matrix& src, int loc_row, int loc_col,
                             int block_size_i, int block_size_j) {
    const int row_range = loc_row + block_size_i;
    int dst_loc = 0;
    for (int i = loc_row; i < row_range; ++i) {
        memcpy(&dst[dst_loc], src[i] + loc_col, block_size_j*(sizeof(int)));
        // printf("%d %d %d %d\n", dst_loc, i, loc_col, block_size_j);
        dst_loc += block_size_j;
    }
}

void inline omp_simd_ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result, int block_size,
                                int i_start, int i_end, int j_start, int j_end) {
#pragma omp parallel shared(M, N, K, matrix1, matrix2, result, block_size) 
{
    int* zeroload_matrix1 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
    int* zeroload_matrix2 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));

    #pragma omp for
    for (int i = i_start; i < i_end; i+=block_size) {
        // printf("Hey, I'm thread %d inside the par zone!\n", omp_get_thread_num()); 
        for (int j = j_start; j < j_end; j+=block_size) {
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


Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, bool row_split,
                             int i_start, int i_end, int j_start, int j_end) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    if (row_split) {
        omp_simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 64, 
                            i_start, i_end, 0, N);    
    } else {
        omp_simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 64, 
                            0, M, j_start, j_end);  
    }
 
    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    // assign tasks
    // const int block_size = 64;
    // const int total_blocks = matrix1.getRows() / block_size * matrix2.getCols() / block_size;
    // int task_blocks = total_blocks / numtasks;
    // int left_blocks = total_block / 
    // int i_range_array[numtasks + 1], j_range_array[numtasks + 1];
    bool row_split = false, col_split = false;
    int *i_cuts, *j_cuts;
    while ((!row_split) && (!col_split)) {
        i_cuts = (int*)calloc((numtasks + 1) , sizeof(int));
        j_cuts = (int*)calloc((numtasks + 1) , sizeof(int));

        if (matrix1.getRows() >= numtasks) {
            assign_cuts(matrix1.getRows(), numtasks, i_cuts);
            row_split = true;
        } else if (matrix2.getCols() >= numtasks) {
            assign_cuts(matrix2.getCols(), numtasks, j_cuts);
            col_split = true;
        } else {
            numtasks = (matrix1.getRows() >= matrix2.getCols()) ? matrix1.getRows() : matrix2.getCols();
        }
    }

    // if (row_split) {
    //     int *j_cuts_t = j_cuts;
    //     for (int i = 0; i < numtasks; ++i) {
    //         *j_cuts_t = 0;
    //         *j_cuts_t = matrix2.getCols()
    //     }
    // }
    

    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        printf("row_split: %d\n", row_split);
        printf("%d i_cuts:", taskid);
        for (int i = 0; i < numtasks + 1; i++) {
            printf("%d ", i_cuts[i]);
        }
        printf("\n");
        printf("%d j_cuts:", taskid);
        for (int i = 0; i < numtasks + 1; i++) {
            printf("%d ", j_cuts[i]);
        }
        printf("\n");

        Matrix result = matrix_multiply_mpi(matrix1, matrix2, row_split,
                                             i_cuts[0], i_cuts[1], j_cuts[0], j_cuts[1]);
        
        // Your Code Here for Synchronization!
        for (int i = MASTER + 1; i < numtasks; i++) {
            int length = 0;
            if (row_split) length = (i_cuts[i + 1] - i_cuts[i]) * matrix2.getCols();
            else length = matrix1.getRows() * (j_cuts[i + 1] - j_cuts[i]);
            int start_pos[length];
            MPI_Recv(start_pos, length, MPI_INT32_T, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            if (row_split) store_res(result, start_pos, i_cuts[i], 0, i_cuts[i + 1] - i_cuts[i], matrix2.getCols());
            else store_res(result, start_pos, 0, j_cuts[i], matrix1.getRows(), j_cuts[i + 1] - j_cuts[i]);
            printf("%d recvd: result %d, totoal: %d\n", status.MPI_SOURCE, start_pos[0], length);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        if (taskid >= numtasks) {
            return 0;
        }
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, row_split,
                                            i_cuts[taskid], i_cuts[taskid + 1],
                                            j_cuts[taskid], j_cuts[taskid + 1]);
        int length = 0;              
        if (row_split) length = (i_cuts[taskid + 1] - i_cuts[taskid]) * matrix2.getCols();
        else length = matrix1.getRows() * (j_cuts[taskid + 1] - j_cuts[taskid]);
        // int length = (i_cuts[taskid + 1] - i_cuts[taskid]) * (j_cuts[taskid + 1] - j_cuts[taskid]);
        int start_pos[length];
        
        // MPI_Recv(start_pos, length, MPI_INT, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        if (row_split) load_res(start_pos, result, i_cuts[taskid], 0, i_cuts[taskid + 1] - i_cuts[taskid], matrix2.getCols());
        else load_res(start_pos, result, 0, j_cuts[taskid], matrix1.getRows(), j_cuts[taskid + 1] - j_cuts[taskid]);
        MPI_Send(start_pos, length, MPI_INT, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // Your Code Here for Synchronization!
        printf("%d sent: result %d~%d(%d), total: %d\n", taskid, start_pos[0], start_pos[length-1], result[i_cuts[taskid + 1] - 1][matrix2.getCols() - 1] ,length);
    }

    MPI_Finalize();
    return 0;
}