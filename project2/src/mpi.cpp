#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <string.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
// #include <iostream>
#include <thread>

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

void inline assign_cuts(int total_workload, int num_tasks, int* cuts) {
    // int total_pixel_num = input_jpeg.width * input_jpeg.height;
    int work_num_per_task = total_workload / num_tasks;
    int left_pixel_num = total_workload % num_tasks;

    // std::vector<int> cuts(num_tasks + 1, 0);
    int divided_left_pixel_num = 0;

    for (int i = 0; i < num_tasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + work_num_per_task + 1;
            divided_left_pixel_num++;
        } else cuts[i+1] = cuts[i] + work_num_per_task;
    }
}

void inline avx_memcpy(void* __restrict dst, const void* __restrict src, int block_size) {
    memcpy(dst, src, block_size*sizeof(int));
}


void inline preload_block(int *__restrict dst, const Matrix& src, int src_row,
                             int src_col, int block_size_row, int block_size_col) {
    for (int i = 0; i < block_size_row; ++i) {
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

void inline omp_simd_ijk_kij_tmm(int M, int N, int K, const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                                int i_start, int i_end, int j_start, int j_end, int thread_num) {
// #pragma omp parallel shared(M, N, K, matrix1, matrix2, result, block_size) 
    int row_num_per_thread = (i_end - i_start) / thread_num;
    int left_row_num = (i_end - i_start) % thread_num;
    int row_cuts[thread_num + 1];
    row_cuts[0] = i_start;
    int divided_left_pixel_num = 0;
    for (int i = 0; i < thread_num; i++) {
        if (divided_left_pixel_num < left_row_num) {
            row_cuts[i+1] = row_cuts[i] + row_num_per_thread + 1;
            divided_left_pixel_num++;
        } else row_cuts[i+1] = row_cuts[i] + row_num_per_thread;
    }

#pragma omp parallel 

{
    // int* zeroload_matrix1 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));
    // int* zeroload_matrix2 = (int*)aligned_alloc(64, block_size * block_size * sizeof(int));

    // #pragma omp for
    int id = omp_get_thread_num();
    // if (row_cuts[id+1] - row_cuts[id] == 0) return;
    const int std_block_size_i = assign_block_size(row_cuts[id+1] - row_cuts[id]);
    // if (std_block_size_i == 0) return;
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(j_end - j_start);


    const int i_res = (row_cuts[id+1] - row_cuts[id]) % std_block_size_i;
    const int k_res = K % std_block_size_k;
    const int j_res = (j_end - j_start) % std_block_size_j;

    const int block_range_i = row_cuts[id+1] - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = j_end - j_res;

    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;
    // printf("M:%d, N:%d, K:%d\n", M, N, K);
    // printf("blk_M:%d, blk_N:%d, blk_K:%d\n", block_size_i, block_size_j, block_size_k);


    int* zeroload_matrix1 = (int*)aligned_alloc(64, (block_size_i * block_size_k + 8) * sizeof(int));
    int* zeroload_matrix2 = (int*)aligned_alloc(64, (block_size_k * block_size_j + 8) * sizeof(int));
    int* kernel_result = (int*)aligned_alloc(64, (block_size_i * block_size_j + 8) * sizeof(int));

    for (int i = row_cuts[id]; i < row_cuts[id+1];) {
        // printf("Hey, I'm thread %d inside the par zone!\n", omp_get_thread_num()); 
        // printf("id: %d i: %d max_i: %d\n", id, i, row_cuts[id+1]);
        for (int j = j_start; j < j_end;) {
            // int kernel_result[block_size * (block_size + 8)] = {};
            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(int));

            for (int k = 0; k < K;) {

                preload_block(zeroload_matrix1, matrix1, i, k, block_size_i, block_size_k);
                preload_block(zeroload_matrix2, matrix2, k, j, block_size_k, block_size_j);
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


Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, bool row_split,
                             int i_start, int i_end, int j_start, int j_end, int thread_num) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    if (row_split) {
        omp_simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result,
                            i_start, i_end, 0, N, thread_num);    
    } else {
        omp_simd_ijk_kij_tmm(M, N, K, matrix1, matrix2, result, 
                            0, M, j_start, j_end, thread_num);  
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
    // printf("numtasks:%d\n", numtasks);
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
    int *i_cuts0, *j_cuts0;
    while ((!row_split) && (!col_split)) {
        i_cuts0 = (int*)calloc((numtasks + 1) , sizeof(int));
        j_cuts0 = (int*)calloc((numtasks + 1) , sizeof(int));

        if (matrix1.getRows() >= numtasks) {
            assign_cuts(matrix1.getRows(), numtasks, i_cuts0);
            row_split = true;
            break;
        } else if (matrix2.getCols() >= numtasks) {
            assign_cuts(matrix2.getCols(), numtasks, j_cuts0);
            col_split = true;
            break;
        }
        numtasks = (matrix1.getRows() >= matrix2.getCols()) ? matrix1.getRows() : matrix2.getCols();
    }
    int i_cuts[numtasks + 1], j_cuts[numtasks + 1];
    if (row_split) {
        for (int i = 0; i < numtasks + 1; ++i) i_cuts[i] = i_cuts0[i];
    } else {
        for (int i = 0; i < numtasks + 1; ++i) j_cuts[i] = j_cuts0[i];
    }


    // int *i_cuts, *j_cuts;
    // while ((!row_split) && (!col_split)) {
    //     i_cuts = (int*)calloc((numtasks + 1) , sizeof(int));
    //     j_cuts = (int*)calloc((numtasks + 1) , sizeof(int));

    //     if (matrix1.getRows() >= numtasks) {
    //         assign_cuts(matrix1.getRows(), numtasks, i_cuts);
    //         row_split = true;
    //     } else if (matrix2.getCols() >= numtasks) {
    //         assign_cuts(matrix2.getCols(), numtasks, j_cuts);
    //         col_split = true;
    //     } else {
    //         numtasks = (matrix1.getRows() >= matrix2.getCols()) ? matrix1.getRows() : matrix2.getCols();
    //     }
    // }


    

    // if (row_split) {
    //     int *j_cuts_t = j_cuts;
    //     for (int i = 0; i < numtasks; ++i) {
    //         *j_cuts_t = 0;
    //         *j_cuts_t = matrix2.getCols()
    //     }
    // }
    

    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        // printf("reassigned numtasks:%d, row_split:%d\n, ", numtasks, row_split);

        // printf("row_split: %d\n", row_split);
        // printf("%d i_cuts:", taskid);
        // for (int i = 0; i < numtasks + 1; i++) {
        //     printf("%d ", i_cuts[i]);
        // }
        // printf("\n");
        // printf("%d j_cuts:", taskid);
        // for (int i = 0; i < numtasks + 1; i++) {
        //     printf("%d ", j_cuts[i]);
        // }
        // printf("\n");

        Matrix result = matrix_multiply_mpi(matrix1, matrix2, row_split,
                                             i_cuts[0], i_cuts[1], j_cuts[0], j_cuts[1], thread_num);
        
        // Your Code Here for Synchronization!
        for (int i = MASTER + 1; i < numtasks; i++) {
            int length = 0;
            if (row_split) length = (i_cuts[i + 1] - i_cuts[i]) * matrix2.getCols();
            else length = matrix1.getRows() * (j_cuts[i + 1] - j_cuts[i]);
            int *start_pos = (int*)malloc(length * sizeof(int));
            MPI_Recv(start_pos, length, MPI_INT32_T, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            if (row_split) store_res(result, start_pos, i_cuts[i], 0, i_cuts[i + 1] - i_cuts[i], matrix2.getCols());
            else store_res(result, start_pos, 0, j_cuts[i], matrix1.getRows(), j_cuts[i + 1] - j_cuts[i]);
            // printf("%d recvd: result %d, totoal: %d\n", status.MPI_SOURCE, start_pos[0], length);
            delete []start_pos;
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

        delete []i_cuts0;
        delete []j_cuts0;

    } else {
        // printf("%d %d\n", i_cuts[taskid], i_cuts[taskid + 1]);

        Matrix result = matrix_multiply_mpi(matrix1, matrix2, row_split,
                                            i_cuts[taskid], i_cuts[taskid + 1],
                                            j_cuts[taskid], j_cuts[taskid + 1], thread_num);
        int length = 0;              
        if (row_split) length = (i_cuts[taskid + 1] - i_cuts[taskid]) * matrix2.getCols();
        else length = matrix1.getRows() * (j_cuts[taskid + 1] - j_cuts[taskid]);
        // int length = (i_cuts[taskid + 1] - i_cuts[taskid]) * (j_cuts[taskid + 1] - j_cuts[taskid]);
        int *start_pos = (int*)malloc(length * sizeof(int));
        
        // MPI_Recv(start_pos, length, MPI_INT, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        // printf("%d %d\n", i_cuts[taskid], i_cuts[taskid + 1]);
        if (row_split) load_res(start_pos, result, i_cuts[taskid], 0, i_cuts[taskid + 1] - i_cuts[taskid], matrix2.getCols());
        else load_res(start_pos, result, 0, j_cuts[taskid], matrix1.getRows(), j_cuts[taskid + 1] - j_cuts[taskid]);
        MPI_Send(start_pos, length, MPI_INT32_T, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        // Your Code Here for Synchronization!
        // printf("%d sent: result %d~%d(%d), total: %d\n", taskid, start_pos[0], start_pos[length-1], result[i_cuts[taskid + 1] - 1][matrix2.getCols() - 1] ,length);
        delete []start_pos;
    }


    MPI_Finalize();

    return 0;
}