#include "simple_ml_ext.hpp"

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(const float *A, size_t m, size_t n)
{
    printf("==========================================\n");
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/*
* assign block sizes
* A helper function for matrix_dot.
*/
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

/*
* load block
* A helper function for matrix_dot.
*/
void inline load_block(float *__restrict dst, const float* src, int src_row, int src_row_len,
                             int src_col, int block_size_row, int block_size_col) {
    // #pragma GCC unroll 64
    for (int i = 0; i < block_size_row; ++i) {
        memcpy(dst, &src[src_row * src_row_len + src_col], block_size_col * sizeof(float));
        dst += block_size_col;
        src_row++;
    }
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t input_k)
{
    int M = m, K = n, N = input_k; 
    const int std_block_size_i = assign_block_size(M);
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(N);
    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;

    const int i_res = M % block_size_i;
    const int k_res = K % block_size_k;
    const int j_res = N % block_size_j;
    const int block_range_i = M - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;

    float* zeroload_matrix1 = (float*)aligned_alloc(64, (block_size_i * block_size_k + 8) * sizeof(float));
    float* zeroload_matrix2 = (float*)aligned_alloc(64, (block_size_k * block_size_j + 8) * sizeof(float));
    float* kernel_result = (float*)aligned_alloc(64, (block_size_i * block_size_j + 8) * sizeof(float));

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
            // memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int temp = 0; temp < block_size_i * block_size_j; ++temp) {
                kernel_result[temp] = 0.0f;
            }
            for (int k = 0; k <= block_range_k;) {
                if (k == K) break;
                if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }
                //------------------kernel----------------------------

                load_block(zeroload_matrix1, A, i, K, k, block_size_i, block_size_k);
                load_block(zeroload_matrix2, B, k, N, j, block_size_k, block_size_j);

                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;  
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        const int r1_iter_loc = (i1 - i) * block_size_k + (k1 - k);
                        register float r1 = *(zeroload_matrix1 + r1_iter_loc);

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
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
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

    // size_t M = m, K = n, N = input_k;

    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         for (size_t k = 0; k < K; ++k) {
    //             C[i * N + j] += A[i * K + k] * B[k * N + j];
    //         }
    //     }
    // }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t m, size_t n, size_t input_k)
{
    // BEGIN YOUR CODE
    int M = m, K = n, N = input_k; 
    const int std_block_size_i = assign_block_size(M);
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(N);
    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;

    const int i_res = M % block_size_i;
    const int k_res = K % block_size_k;
    const int j_res = N % block_size_j;
    const int block_range_i = M - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;

    float* zeroload_matrix1 = (float*)aligned_alloc(64, (block_size_i * block_size_k + 8) * sizeof(float));
    float* zeroload_matrix2 = (float*)aligned_alloc(64, (block_size_k * block_size_j + 8) * sizeof(float));
    float* kernel_result = (float*)aligned_alloc(64, (block_size_i * block_size_j + 8) * sizeof(float));

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
            // memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int temp = 0; temp < block_size_i * block_size_j; ++temp) {
                kernel_result[temp] = 0.0f;
            }
            for (int k = 0; k <= block_range_k;) {
                if (k == K) break;
                if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }
                //------------------kernel----------------------------

                load_block(zeroload_matrix1, A, k, M, i, block_size_k, block_size_i);
                load_block(zeroload_matrix2, B, k, N, j, block_size_k, block_size_j);

                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;  
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        const int r1_iter_loc = (k1 - k) * block_size_i + (i1 - i);
                        register float r1 = *(zeroload_matrix1 + r1_iter_loc);

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
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
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
    // END YOUR CODE
    // size_t M = m, K = n, N = input_k;

    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         for (size_t k = 0; k < K; ++k) {
    //             C[i * N + j] += A[k * M + i] * B[k * N + j];
    //         }
    //     }
    // }
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    int M = m, K = n, N = k; 
    const int std_block_size_i = assign_block_size(M);
    const int std_block_size_k = assign_block_size(K);
    const int std_block_size_j = assign_block_size(N);
    int block_size_i = std_block_size_i, block_size_j = std_block_size_j, block_size_k = std_block_size_k;

    const int i_res = M % block_size_i;
    const int k_res = K % block_size_k;
    const int j_res = N % block_size_j;
    const int block_range_i = M - i_res;
    const int block_range_k = K - k_res;
    const int block_range_j = N - j_res;
    bool i_switch = false;
    bool j_switch = false;
    bool k_switch = false;

    float* zeroload_matrix1 = (float*)aligned_alloc(64, (block_size_i * block_size_k + 8) * sizeof(float));
    float* zeroload_matrix2 = (float*)aligned_alloc(64, (block_size_k * block_size_j + 8) * sizeof(float));
    float* kernel_result = (float*)aligned_alloc(64, (block_size_i * block_size_j + 8) * sizeof(float));

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
            // memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int temp = 0; temp < block_size_i * block_size_j; ++temp) {
                kernel_result[temp] = 0.0f;
            }
            for (int k = 0; k <= block_range_k;) {
                if (k == K) break;
                if (k == block_range_k) {
                    block_size_k = k_res;
                    k_switch = true;
                }
                //------------------kernel----------------------------

                load_block(zeroload_matrix1, A, i, K, k, block_size_i, block_size_k);
                load_block(zeroload_matrix2, B, j, K, k, block_size_j, block_size_k);


                for (int j1 = j; j1 < j + block_size_j; ++j1) {

                    const int temp_kloc_r2 = (j1 - j) * block_size_k;

                    for (int i1 = i; i1 < i+block_size_i; ++i1) {

                        const int result_iter_loc = (i1 - i) * block_size_j;
                        const int temp_kloc_r1 = (i1 - i) * block_size_k;  

                        for (int k1 = k; k1 < k+block_size_k; ++k1) {
                            const int r1_iter_loc = temp_kloc_r1 + (k1 - k);
                            const int r2_iter_loc = temp_kloc_r2 + (k1 - k);
                            kernel_result[result_iter_loc + j1 - j] += zeroload_matrix1[r1_iter_loc] * zeroload_matrix2[r2_iter_loc];     
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
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
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
    // END YOUR CODE
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] -= B[i * n + j]; 
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] *= scalar; 
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] /= scalar; 
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < m; ++i) {
        float row_sum = 0.0f; 
        const int rowloc = n * i;
        for (int j = 0; j < n; ++j) {
            float cur_e = exp(C[rowloc + j]);
            C[rowloc + j] = cur_e;
            row_sum += cur_e;
        }

        for (int j = 0; j < n; ++j) {
            C[rowloc + j] /= row_sum;
        }

    }
    // END YOUR CODE
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < m; ++i) {
        // for (int j = 0; j < n; ++j) {
        //     Y[i * n + j] = (y[i] == j) ? (1) : (0); 
        // }
        Y[i * n + y[i]] = 1;
    }  
    // END YOUR CODE
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): SGD minibatch size
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE 
    for (int i = 0; i < m; i+=batch) {
        const unsigned char* cur_y = y + i;
        const float* X_b = X + i * n; 
        float Z[batch * k] = {};
        float gd[n * k] = {};
        matrix_dot(X_b, theta, Z, batch, n, k); 
        matrix_softmax_normalize(Z, batch, k);

        float Y[batch * k] = {};
        vector_to_one_hot_matrix(cur_y, Y, batch, k);
        
        matrix_minus(Z, Y, batch, k);
        matrix_dot_trans(X_b, Z, gd, n, batch, k); // n*100 * 100*k
        float batchxlr = lr / batch;
        matrix_mul_scalar(gd, batchxlr, n, k);
        matrix_minus(theta, gd, n, k);
    }
    // int itr = m / batch;
    // for (int i0 = 0; i0 < itr; i0++) {
    //     int start = i0 * batch;
    //     const float *cur_X = X + start * int(n); 
    //     const unsigned char *cur_y = y + start; 
    //     // float *Z;
    //     // Z = new float[batch * (int)k];
    //     float Z[batch * (int)k] = {};
    //     matrix_dot(cur_X, theta, Z, batch, n, k);
    //     matrix_softmax_normalize(Z, batch, k);
    //     // matrix::mMul(cur_X, theta, Z, batch, n, k);
    //     // matrix::mExp(Z, batch, k);
    //     // matrix::mNormRow(Z, batch, k);
    //     // float *Iy;
    //     // Iy = new float[batch * (int)k]();
    //     float Iy[batch * (int)k] = {};
    //     for (int i = 0; i < batch; ++i) {
    //         Iy[i * (int)k + cur_y[i]] = 1;
    //     }
    //     // float* xT;
    //     // xT = new float[batch * (int)n];
    //     // matrix::mT(cur_X, xT, (int)n, batch);
    //     // float* gradient;
    //     // gradient = new float[(int)n * (int)k];
    //     float gradient[(int)n * (int)k] = {};
    //     matrix_minus(Z, Iy, batch, k);
    //     // matrix::mSub(Z, Iy, Z, batch, k);
    //     matrix_dot_trans(cur_X, Z, gradient, n, batch, k);
    //     // matrix::mMul(xT, Z, gradient, (int)n, batch, (int)k);
    //     matrix_mul_scalar(gradient, lr / batch, n, k);
    //     // matrix::mMul(gradient, lr / batch, (int)n, (int)k);
    //     matrix_minus(theta, gradient, n, k);
    //     // matrix::mSub(theta, gradient, theta, (int)n, (int)k);
    // }
    // END YOUR CODE
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE
        memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
        memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
        int m = train_data->images_num;
        int n = train_data->input_dim;
        int k = num_classes;
        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, m, n, k, lr, batch);
        // print_matrix(theta, n, k);
        matrix_dot(train_data->images_matrix, theta, train_result, m, n ,k);
        matrix_dot(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim ,k);

        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float res = 0.0f;
    for (int i = 0; i < images_num; ++i) {
        int row_loc = i * num_classes;
        float row_correct = result[row_loc + labels_array[i]];
        float row_exp_sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            row_exp_sum += exp(result[row_loc + j]);
        }
        res += -row_correct + log(row_exp_sum);
    }
    return res / images_num;
    // END YOUR CODE
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float res = 0.0f;
    for (int i = 0; i < images_num; ++i) {
        int row_idx = i * num_classes;
        unsigned char row_max_idx = 0;
        float row_max = result[row_idx];
        for (int j = 1; j < num_classes; ++j) {
            float cur = result[row_idx + j];
            if (cur > row_max) {
                row_max = cur;
                row_max_idx = j;
            }
        }
        res += (row_max_idx == labels_array[i]) ? (0) : (1);
    }
    return res / images_num;
    // END YOUR CODE
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] *= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    for (int i = 0; i < size; ++i) {
        A[i] *= B[i];
    }  
    // END YOUR CODE
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
