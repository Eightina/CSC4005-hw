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
inline void assign_block_size(int M, int K, int N, int* res) {
    res[0] = M;
    res[1] = K;
    res[2] = N;
    // for (int i = 0; i < 3; ++i) {
    while (res[0] % 2 == 0 && res[0] > 128) {
        res[0] /= 2;
    }
    while (res[0] % 3 == 0 && res[0] > 128) {
        res[0] /= 3;
    }
    while (res[0] % 5 == 0 && res[0] > 128) {
        res[0] /= 5;
    }

    while (res[1] % 2 == 0 && res[1] > 64) {
        res[1] /= 2;
    }
    while (res[1] % 3 == 0 && res[1] > 64) {
        res[1] /= 3;
    }
    while (res[1] % 5 == 0 && res[1] > 64) {
        res[1] /= 5;
    }
        // while (res[i] % 7 == 0 && res[i] > 64) {
        //     res[i] /= 7;
        // }
        // while (res[i] % 11 == 0 && res[i] > 64) {
        //     res[i] /= 11;
        // }
    // }
    // printf("M %d K %d N %d\n", res[0], res[1], res[2]);
    return;

    // strategy 1
    // int max_loc = 0;
    // int max = M;
    // if (K > max) {
    //     max_loc = 1;
    //     max = K;
    // }
    // if (N > max) {
    //     max_loc = 2;
    //     max = N;
    // }
    // res[0] = M;
    // res[1] = K;
    // res[2] = N;
    // while (max % 2 == 0 && max > 64) {
    //     max /= 2;
    // }
    // res[max_loc] = max;

    // strategy 2
    // if (M >= 64) {
    //     return 64;
    // } else if (M >= 32) {
    //     return 32;
    // } else if (M >= 16) {
    // return 16;
    // } else if (M >= 8) {
    // return 8;
    // } else if (M >= 4) {
    // return 4;
    // } 
    // return 1;
}
/*
* assign block sizes
* A helper function for matrix_dot.
*/
void inline assign_blocks(int M, int K, int N, int* block_sizes, float** assist_sapces) {
    assign_block_size(M, K, N, block_sizes);
    assist_sapces[0] = (float*)aligned_alloc(64, (block_sizes[0] * block_sizes[1] + 8) * sizeof(float));
    assist_sapces[1] = (float*)aligned_alloc(64, (block_sizes[1] * block_sizes[2] + 8) * sizeof(float));
    assist_sapces[2] = (float*)aligned_alloc(64, (block_sizes[0] * block_sizes[2] + 8) * sizeof(float));
}
/*
* load block
* A helper function for matrix_dot.
*/
void inline load_block(float *__restrict dst, const float*__restrict src, int src_row, int src_row_len,
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
void matrix_dot(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t input_k, int* block_sizes, float** assist_spaces)
{
    int M = m, K = n, N = input_k; 
    int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];

    float* __restrict zeroload_matrix1 = assist_spaces[0];
    float* __restrict zeroload_matrix2 = assist_spaces[1];
    float* __restrict kernel_result = assist_spaces[2];

    for (int i = 0; i < M; i += block_size_i) {
        for (int j = 0; j < N; j += block_size_j) {
            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int k = 0; k < K; k += block_size_k) {

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

            }
            for (int row = 0; row < block_size_i; ++row) {
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t input_k, int* block_sizes, float** assist_spaces)
{
    // BEGIN YOUR CODE
    int M = m, K = n, N = input_k; 
    int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];

    float* __restrict zeroload_matrix1 = assist_spaces[0];
    float* __restrict zeroload_matrix2 = assist_spaces[1];
    float* __restrict kernel_result = assist_spaces[2];

    for (int i = 0; i < M; i += block_size_i) {
        for (int j = 0; j < N; j += block_size_j) {
            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int k = 0; k < K; k += block_size_k) {

                //------------------kernel----------------------------
                load_block(zeroload_matrix1, A, k, M, i, block_size_k, block_size_i);
                load_block(zeroload_matrix2, B, k, N, j, block_size_k, block_size_j);
                for (int k1 = k; k1 < k+block_size_k; ++k1) {
                    const int temp_kloc = (k1 - k) * block_size_j;
                    float r1_row[block_size_i];
                    memcpy(r1_row, zeroload_matrix1 + (k1 - k) * block_size_i, block_size_i * sizeof(float));
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        const int result_iter_loc = (i1 - i) * block_size_j;
                        for (int j1 = j; j1 < j + block_size_j; ++j1) {
                            kernel_result[result_iter_loc + j1 - j] += r1_row[i1 - i] * zeroload_matrix2[temp_kloc + j1 - j];  
                        }
                    }
                }
                //------------------kernel----------------------------

            }
            for (int row = 0; row < block_size_i; ++row) {
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t input_k, int* block_sizes, float** assist_spaces)
{
    int M = m, K = n, N = input_k; 
    int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];
    
    float* __restrict zeroload_matrix1 = assist_spaces[0];
    float* __restrict zeroload_matrix2 = assist_spaces[1];
    float* __restrict kernel_result = assist_spaces[2];

    for (int i = 0; i < M; i += block_size_i) {
        for (int j = 0; j < N; j += block_size_j) {
            memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));
            for (int k = 0; k < K; k += block_size_k) {

                //------------------kernel----------------------------
                load_block(zeroload_matrix1, A, i, K, k, block_size_i, block_size_k);
                load_block(zeroload_matrix2, B, j, K, k, block_size_j, block_size_k);
                for (int j1 = j; j1 < j + block_size_j; ++j1) {
                    const int temp_kloc_r2 = (j1 - j) * block_size_k;
                    for (int i1 = i; i1 < i+block_size_i; ++i1) {
                        const int result_iter_loc = (i1 - i) * block_size_j;
                        const int temp_kloc_r1 = (i1 - i) * block_size_k;  
                        for (int k1 = k; k1 < k+block_size_k; ++k1) {
                            kernel_result[result_iter_loc + j1 - j] += zeroload_matrix1[temp_kloc_r1 + (k1 - k)] * zeroload_matrix2[temp_kloc_r2 + (k1 - k)];     
                        }
                    }
                }
                //------------------kernel----------------------------

            }
            for (int row = 0; row < block_size_i; ++row) {
                memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
            }
        }
    }
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *__restrict A, const float *__restrict B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    size_t range = m * n;
    while (range > 0) {
        (*A) -= (*B);
        --range;
        ++A;
        ++B;
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
void matrix_mul_scalar(float *__restrict C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    size_t range = m * n;
    while (range > 0) {
        (*C) *= scalar;
        --range;
        ++C;
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
void matrix_div_scalar(float *__restrict C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    scalar = 1 / scalar;
    size_t range = m * n;
    while (range > 0) {
        (*C) *= scalar;
        --range;
        ++C;
    }
    // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *__restrict C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // float curC[n];
    for (size_t i = 0; i < m; ++i) {
        const size_t rowloc = n * i;
        float* curC = C + rowloc;
        // memcpy(curC, C + rowloc, sizeof(float) * n);

        float row_sum = 0.0f; 
        for (size_t j = 0; j < n; ++j) {
            float cur_e = exp(curC[j]);
            curC[j] = cur_e;
            row_sum += cur_e;
        }
                // use mul
        row_sum = 1 / row_sum;
        for (size_t j = 0; j < n; ++j) {
            curC[j] *= row_sum;
        }

        // memcpy(C + rowloc, curC, sizeof(float) * n);

    }
    // END YOUR CODE
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *__restrict y, float *__restrict Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    while (m > 0) {
        *(Y + (*y)) = 1;
        --m;
        ++y;
        Y += n;
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
void softmax_regression_epoch_cpp(const float *__restrict X, const unsigned char *__restrict y, float *__restrict theta,
                                 size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE 
    const float batchxlr = lr / batch;
    unsigned char cur_y[batch];
    float X_b[batch * n];
    float Z[batch * k];
    float gd[n * k];
    float Y[batch * k];

    // assitant matrix space in dot 
    int block_sizes[3];
    int block_sizes_t[3];
    float* assist_sapces[3];
    float* assist_sapces_t[3];
    assign_blocks(batch, n, k, block_sizes, assist_sapces);
    assign_blocks(n, batch, k, block_sizes_t, assist_sapces_t);


    for (int i = 0; i < m; i += batch) {
        memset(Z, 0, batch * k * sizeof(float));
        memset(gd, 0, n * k * sizeof(float));
        memset(Y, 0, batch * k * sizeof(float));

        memcpy(cur_y, y + i, batch * sizeof(unsigned char)); 
        memcpy(X_b, X + i * n, batch * n * sizeof(float));

        matrix_dot(X_b, theta, Z, batch, n, k, block_sizes, assist_sapces); 
        matrix_softmax_normalize(Z, batch, k);

        vector_to_one_hot_matrix(cur_y, Y, batch, k);
        
        matrix_minus(Z, Y, batch, k);
        matrix_dot_trans(X_b, Z, gd, n, batch, k, block_sizes_t, assist_sapces_t); // n*100 * 100*k
        matrix_mul_scalar(gd, batchxlr, n, k);
        matrix_minus(theta, gd, n, k);
    }
    for (int i = 0; i < 3; ++i) {
        free(assist_sapces[i]);
        free(assist_sapces_t[i]);
    }
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

    // BEGIN YOUR CODE
    // assign space for matrix dot
    const int m_train = train_data->images_num;
    const int n_train = train_data->input_dim;
    const int m_test = test_data->images_num;
    const int n_test = test_data->input_dim;
    const int k = num_classes;

    int block_sizes_train[3];
    int block_sizes_test[3];
    float* assist_sapces_train[3];
    float* assist_sapces_test[3];
    assign_blocks(m_train, n_train, k, block_sizes_train, assist_sapces_train);
    assign_blocks(m_test, n_test, k, block_sizes_test, assist_sapces_test);

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {

        memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
        memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));

        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta,
                                        m_train, n_train, k, lr, batch);
        matrix_dot(train_data->images_matrix, theta, train_result, m_train, n_train ,k, block_sizes_train, assist_sapces_train);
        matrix_dot(test_data->images_matrix, theta, test_result, m_test, n_test ,k, block_sizes_test, assist_sapces_test);

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

    for (int i = 0; i < 3; ++i) {
        free(assist_sapces_train[i]);
        free(assist_sapces_test[i]);
    }
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
float mean_softmax_loss(const float *__restrict result, const unsigned char *__restrict labels_array, size_t images_num, size_t num_classes)
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
float mean_err(float *__restrict result, const unsigned char *__restrict labels_array, size_t images_num, size_t num_classes)
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
void matrix_mul(float *__restrict A, const float *__restrict B, size_t size) {    
    while (size > 0) {
        (*A) *= (*B);
        ++A;
        ++B;
        --size;
    }
}

/**
 * A helper function: Masking Matrix 
 * Efficiently compute A = A * (B > 0)
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Masking Matrix of size m * n
 **/
void matrix_masking(float *__restrict A, const float *__restrict B, size_t size) {    
    while (size > 0) {
        (*A) = ((*B) > 0) ? (*A) : 0;
        ++A;
        ++B;
        --size;
    }
}

/**
 * A helper function for ReLu 
 * Efficiently compute A = ReLu(A)
 * Args:
 *     A (const float*): Matrix of size m * n
 **/
void relu(float *__restrict A, size_t m, size_t n) {
    size_t range = m * n;
    while (range > 0) {
        (*A) = ((*A) > 0) ? (*A) : 0;
        --range;
        ++A;
    }
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
        (input_dim x hidden_dim, n x l)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes, l x k)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch
*/
void nn_epoch_cpp(const float *__restrict X, const unsigned char *__restrict y, float *__restrict W1, float *__restrict W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float X_b[batch * n];
    float Z1[batch * l];
    float G1[batch * l];
    // float h_Z1_exp[batch * k];
    float Z2[batch * k];
    float Y[batch * k];
    float W1_l[n * l];
    float W2_l[l * k];
    unsigned char cur_y[batch];

    int block_sizes[15];
    // int block_sizes_test[3];
    float* assist_sapces[15];
    // float* assist_sapces_test[3];
    assign_blocks(batch, n, l, block_sizes, assist_sapces);
    assign_blocks(batch, l, k, block_sizes + 3, assist_sapces + 3);
    assign_blocks(batch, k, l, block_sizes + 6, assist_sapces + 6);
    assign_blocks(n, batch, l, block_sizes + 9, assist_sapces + 9);
    assign_blocks(l, batch, k, block_sizes + 12, assist_sapces + 12);

    for (int i = 0; i < m; i += batch) {
        memset(Z1, 0, batch * l * sizeof(float));
        memset(Z2, 0, batch * k * sizeof(float));
        memset(G1, 0, batch * l * sizeof(float));
        memset(Y, 0, batch * k * sizeof(float));
        memset(W1_l, 0, n * l * sizeof(float));
        memset(W2_l, 0, l * k * sizeof(float));
        memcpy(cur_y, y + i, batch * sizeof(unsigned char)); 
        memcpy(X_b, X + i * n, batch * n * sizeof(float));

        matrix_dot(X_b, W1, Z1, batch, n, l, block_sizes, assist_sapces); //
        relu(Z1, batch, l);
        matrix_dot(Z1, W2, Z2, batch, l, k, block_sizes + 3, assist_sapces + 3); //
        matrix_softmax_normalize(Z2, batch, k);

        vector_to_one_hot_matrix(cur_y, Y, batch, k);
        matrix_minus(Z2, Y, batch, k); // Z2 = Z2 - Y
        matrix_trans_dot(Z2, W2, G1, batch, k, l, block_sizes + 6, assist_sapces + 6); //
        matrix_masking(G1, Z1, batch * l);

        matrix_dot_trans(X_b, G1, W1_l, n, batch, l, block_sizes + 9, assist_sapces + 9); //
        matrix_dot_trans(Z1, Z2, W2_l, l, batch, k, block_sizes + 12, assist_sapces + 12); //
        matrix_mul_scalar(W1_l, lr / batch, n, l);
        matrix_mul_scalar(W2_l, lr / batch, l, k);
        matrix_minus(W1, W1_l, n, l);
        matrix_minus(W2, W2_l, l, k);
        // print_matrix(W1, n, l);
        // print_matrix(W2, l, k);

    }
    // END YOUR CODE
    for (int i = 0; i < 15; ++i) {
        free(assist_sapces[i]);
    }
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
    
    // BEGIN YOUR CODE
    const int m_train = train_data->images_num;
    const int m_test = test_data->images_num;
    const int n = train_data->input_dim;
    const int k = num_classes;
    const int l = hidden_dim;
    float *train_temp = new float[m_train * l];
    float *test_temp = new float[m_test * l];

    int block_sizes[12];
    float* assist_sapces[12];
    assign_blocks(m_train, n, l, block_sizes, assist_sapces);
    assign_blocks(m_train, l, k, block_sizes + 3, assist_sapces + 3);
    assign_blocks(m_test, n, l, block_sizes + 6, assist_sapces + 6);
    assign_blocks(m_test, l, k, block_sizes + 9, assist_sapces + 9);

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        memset(train_temp, 0, m_train * l * sizeof(float));
        memset(train_result, 0, m_train * k * sizeof(float));
        memset(test_temp, 0, m_test * l * sizeof(float));
        memset(test_result, 0, m_test * k * sizeof(float));

        nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2, m_train, n, l, k, lr, batch);

        matrix_dot(train_data->images_matrix, W1, train_temp, m_train, n ,l, block_sizes, assist_sapces);
        relu(train_temp, m_train, l);
        matrix_dot(train_temp, W2, train_result, m_train, l ,k, block_sizes + 3, assist_sapces + 3);
        
        matrix_dot(test_data->images_matrix, W1, test_temp, m_test, n ,l, block_sizes + 6, assist_sapces + 6);
        relu(test_temp, m_test, l);
        matrix_dot(test_temp, W2, test_result, m_test, l ,k, block_sizes + 9, assist_sapces + 9);
        
        // print_matrix(W1, n, l);
        // print_matrix(W2, l, k);

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
    delete[] train_temp;
    delete[] test_temp;
    for (int i = 0; i < 12; ++i) {
        free(assist_sapces[i]);
    }
}
