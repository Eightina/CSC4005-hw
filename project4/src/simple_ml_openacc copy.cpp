// #include "simple_ml_openacc.hpp"


// /*
// * assign block sizes
// * A helper function for matrix_dot.
// */
// inline void assign_block_size(int M, int K, int N, int* res) {
//     res[0] = M;
//     res[1] = K;
//     res[2] = N;

//     while (res[0] % 2 == 0 && res[0] > 128) {
//         res[0] /= 2;
//     }
//     while (res[0] % 3 == 0 && res[0] > 128) {
//         res[0] /= 3;
//     }
//     while (res[0] % 5 == 0 && res[0] > 128) {
//         res[0] /= 5;
//     }

//     while (res[1] % 2 == 0 && res[1] > 64) {
//         res[1] /= 2;
//     }
//     while (res[1] % 3 == 0 && res[1] > 64) {
//         res[1] /= 3;
//     }
//     while (res[1] % 5 == 0 && res[1] > 64) {
//         res[1] /= 5;
//     }
//     return;
// }
// /*
// * assign block sizes
// * A helper function for matrix_dot.
// */
// void inline assign_blocks(int M, int K, int N, int* block_sizes, float** assist_sapces) {
//     assign_block_size(M, K, N, block_sizes);
//     assist_sapces[0] = (float*)aligned_alloc(64, (block_sizes[0] * block_sizes[1] + 8) * sizeof(float));
//     assist_sapces[1] = (float*)aligned_alloc(64, (block_sizes[1] * block_sizes[2] + 8) * sizeof(float));
//     assist_sapces[2] = (float*)aligned_alloc(64, (block_sizes[0] * block_sizes[2] + 8) * sizeof(float));
// }
// /*
// * load block
// * A helper function for matrix_dot.
// */
// void inline load_block(float *__restrict dst, const float*__restrict src, int src_row, int src_row_len,
//                              int src_col, int block_size_row, int block_size_col) {
//     #pragma acc loop independent 
//     for (int i = 0; i < block_size_row; ++i) {
//         memcpy(dst, &src[src_row * src_row_len + src_col], block_size_col * sizeof(float));
//         dst += block_size_col;
//         src_row++;
//     }
// }

// void matrix_dot_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k, int* block_sizes, float** assist_spaces) 
// {
//     int M = m, K = n, N = k; 
//     int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];
//     int z_m1_size = block_size_i * block_size_k;
//     int z_m2_size = block_size_k * block_size_j;
//     int k_r_size = block_size_i * block_size_j;

//     float* __restrict zeroload_matrix1 = assist_spaces[0];
//     float* __restrict zeroload_matrix2 = assist_spaces[1];
//     float* __restrict kernel_result = assist_spaces[2];
//     #pragma acc enter data create(kernel_result[0:k_r_size], zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size])

//     #pragma acc parallel present(A, B, C, \
//         zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])\
//          num_gangs(1024) 
//     {
//     #pragma acc loop independent
//     for (int i = 0; i < M; i += block_size_i) {

//         #pragma acc loop independent
//         for (int j = 0; j < N; j += block_size_j) {
//             memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));

//             #pragma acc loop independent
//             for (int k = 0; k < K; k += block_size_k) {

//                 //------------------kernel----------------------------
//                 load_block(zeroload_matrix1, A, i, K, k, block_size_i, block_size_k);
//                 load_block(zeroload_matrix2, B, k, N, j, block_size_k, block_size_j);

//                 #pragma acc loop independent
//                 for (int k1 = k; k1 < k+block_size_k; ++k1) {
//                     const int temp_kloc = (k1 - k) * block_size_j;

//                     #pragma acc loop independent
//                     for (int i1 = i; i1 < i+block_size_i; ++i1) {
//                         const int r1_iter_loc = (i1 - i) * block_size_k + (k1 - k);
//                         register float r1 = *(zeroload_matrix1 + r1_iter_loc);
//                         const int result_iter_loc = (i1 - i) * block_size_j;

//                         // #pragma acc loop independent
//                         for (int j1 = j; j1 < j + block_size_j; ++j1) {
//                             kernel_result[result_iter_loc + j1 - j] += r1 * zeroload_matrix2[temp_kloc + j1 - j];  
//                         }
//                     }
//                 }
//                 //------------------kernel----------------------------

//             }
//             #pragma acc loop independent
//             for (int row = 0; row < block_size_i; ++row) {
//                 memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
//             }
//         }
//     }
//     }
//     #pragma acc exit data delete(zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])
// }
// /**
//  * Matrix Dot Multiplication Trans Version
//  * Efficiently compute C = A.T.dot(B)
//  * Args:
//  *     A (const float*): Matrix of size n * m
//  *     B (const float*): Matrix of size n * k
//  *     C (float*): Matrix of size m * k
//  **/
// void matrix_dot_trans_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t n, size_t m, size_t k, int* block_sizes, float** assist_spaces) 
// {
//     int M = m, K = n, N = k; 
//     int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];
//     int z_m1_size = block_size_i * block_size_k;
//     int z_m2_size = block_size_k * block_size_j;
//     int k_r_size = block_size_i * block_size_j;

//     float* __restrict zeroload_matrix1 = assist_spaces[0];
//     float* __restrict zeroload_matrix2 = assist_spaces[1];
//     float* __restrict kernel_result = assist_spaces[2];
//     #pragma acc enter data create(kernel_result[0:k_r_size], zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size])

//     #pragma acc parallel present(A, B, C, \
//         zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])\
//          num_gangs(1024) 
//     {
//     #pragma acc loop independent
//     for (int i = 0; i < M; i += block_size_i) {

//         #pragma acc loop independent
//         for (int j = 0; j < N; j += block_size_j) {
//             memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));

//             #pragma acc loop independent
//             for (int k = 0; k < K; k += block_size_k) {

//                 //------------------kernel----------------------------
//                 load_block(zeroload_matrix1, A, k, M, i, block_size_k, block_size_i);
//                 load_block(zeroload_matrix2, B, k, N, j, block_size_k, block_size_j);

//                 #pragma acc loop independent
//                 for (int k1 = k; k1 < k+block_size_k; ++k1) {
//                     const int temp_kloc = (k1 - k) * block_size_j;
//                     float r1_row[block_size_i];
//                     memcpy(r1_row, zeroload_matrix1 + (k1 - k) * block_size_i, block_size_i * sizeof(float));

//                     #pragma acc loop independent
//                     for (int i1 = i; i1 < i+block_size_i; ++i1) {
//                         const int result_iter_loc = (i1 - i) * block_size_j;

//                         // #pragma acc parallel loop independent
//                         for (int j1 = j; j1 < j + block_size_j; ++j1) {
//                             kernel_result[result_iter_loc + j1 - j] += r1_row[i1 - i] * zeroload_matrix2[temp_kloc + j1 - j];  
//                         }
//                     }
//                 }
//                 //------------------kernel----------------------------

//             }
//             #pragma acc loop independent
//             for (int row = 0; row < block_size_i; ++row) {
//                 memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
//             }
//         }
//     }
//     }
//     #pragma acc exit data delete(zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])
// }
// /**
//  * Matrix Dot Multiplication Trans Version 2
//  * Efficiently compute C = A.dot(B.T)
//  * Args:
//  *     A (const float*): Matrix of size m * n
//  *     B (const float*): Matrix of size k * n
//  *     C (float*): Matrix of size m * k
//  **/
// void matrix_trans_dot_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k, int* block_sizes, float** assist_spaces)
// {
//     int M = m, K = n, N = k; 
//     int block_size_i = block_sizes[0], block_size_k = block_sizes[1], block_size_j = block_sizes[2];
//     int z_m1_size = block_size_i * block_size_k;
//     int z_m2_size = block_size_k * block_size_j;
//     int k_r_size = block_size_i * block_size_j;

//     float* __restrict zeroload_matrix1 = assist_spaces[0];
//     float* __restrict zeroload_matrix2 = assist_spaces[1];
//     float* __restrict kernel_result = assist_spaces[2];
//     #pragma acc enter data create(kernel_result[0:k_r_size], zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size])

//     #pragma acc parallel present(A, B, C, \
//         zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])\
//          num_gangs(1024) 
//     {
//     #pragma acc loop independent
//     for (int i = 0; i < M; i += block_size_i) {

//         #pragma acc loop independent
//         for (int j = 0; j < N; j += block_size_j) {
//             memset(kernel_result, 0, block_size_i * block_size_j * sizeof(float));

//             #pragma acc loop independent
//             for (int k = 0; k < K; k += block_size_k) {

//                 //------------------kernel----------------------------
//                 load_block(zeroload_matrix1, A, i, K, k, block_size_i, block_size_k);
//                 load_block(zeroload_matrix2, B, j, K, k, block_size_j, block_size_k);

//                 #pragma acc loop independent
//                 for (int j1 = j; j1 < j + block_size_j; ++j1) {
//                     const int temp_kloc_r2 = (j1 - j) * block_size_k;

//                     #pragma acc loop independent
//                     for (int i1 = i; i1 < i+block_size_i; ++i1) {
//                         const int result_iter_loc = (i1 - i) * block_size_j;
//                         const int temp_kloc_r1 = (i1 - i) * block_size_k;  

//                         // #pragma acc parallel loop independent
//                         for (int k1 = k; k1 < k+block_size_k; ++k1) {
//                             kernel_result[result_iter_loc + j1 - j] += zeroload_matrix1[temp_kloc_r1 + (k1 - k)] * zeroload_matrix2[temp_kloc_r2 + (k1 - k)];     
//                         }
//                     }
//                 }
//                 //------------------kernel----------------------------

//             }
//             #pragma acc loop independent
//             for (int row = 0; row < block_size_i; ++row) {
//                 memcpy(&C[(i + row) * N + j], &kernel_result[row * block_size_j], block_size_j * sizeof(float));
//             }
//         }
//     }
//     }
//     #pragma acc exit data delete(zeroload_matrix1[0:z_m1_size], zeroload_matrix2[0:z_m2_size], kernel_result[0:k_r_size])
// }

// void matrix_minus_openacc(float *__restrict A, const float *__restrict B, size_t m, size_t n)
// {
//     size_t range = m * n;
//     #pragma acc loop independent
//     while (range > 0) {
//         (*A) -= (*B);
//         --range;
//         ++A;
//         ++B;
//     }
// }

// void matrix_mul_scalar_openacc(float *__restrict C, float scalar, size_t m, size_t n)
// {
//     // BEGIN YOUR CODE
//     size_t range = m * n;
//     #pragma acc loop independent
//     while (range > 0) {
//         (*C) *= scalar;
//         --range;
//         ++C;
//     }
//     // END YOUR CODE
// }

// void matrix_div_scalar_openacc(float *__restrict C, float scalar, size_t m, size_t n)
// {
//     // BEGIN YOUR CODE
//     scalar = 1 / scalar;
//     size_t range = m * n;
//     #pragma acc loop independent
//     while (range > 0) {
//         (*C) *= scalar;
//         --range;
//         ++C;
//     }
//     // END YOUR CODE
// }

// void matrix_softmax_normalize_openacc(float *__restrict C, size_t m, size_t n)
// {
//     #pragma acc loop independent
//     for (size_t i = 0; i < m; ++i) {
//         const size_t rowloc = n * i;
//         float* curC = C + rowloc;

//         float row_sum = 0.0f; 
//         for (size_t j = 0; j < n; ++j) {
//             float cur_e = exp(curC[j]);
//             curC[j] = cur_e;
//             row_sum += cur_e;
//         }
//         // use mul
//         row_sum = 1 / row_sum;
//         for (size_t j = 0; j < n; ++j) {
//             curC[j] *= row_sum;
//         }
//     }
// }

// void vector_to_one_hot_matrix_openacc(const unsigned char *__restrict y, float *__restrict Y, size_t m, size_t n)
// {
//     #pragma acc loop independent
//     while (m > 0) {
//         *(Y + (*y)) = 1;
//         --m;
//         ++y;
//         Y += n;
//     }
// }

//    // BEGIN YOUR CODE 
// void softmax_regression_epoch_openacc(const float *__restrict X, const unsigned char *__restrict y,
//                                       float *__restrict theta, size_t m, size_t n, size_t k,
//                                       float lr, size_t batch)
// {
//     const float batchxlr = lr / batch;
//     unsigned char cur_y[batch];
//     float X_b[batch * n];
//     float Z[batch * k];
//     float gd[n * k];
//     float Y[batch * k];
//     #pragma acc enter data create(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 

//     // assitant matrix space in dot 
//     int block_sizes[3];
//     int block_sizes_t[3];
//     float* assist_sapces[3];
//     float* assist_sapces_t[3];
//     assign_blocks(batch, n, k, block_sizes, assist_sapces);
//     assign_blocks(n, batch, k, block_sizes_t, assist_sapces_t);

//     for (int i = 0; i < m; i += batch) {
//         memset(Z, 0, batch * k * sizeof(float));
//         memset(gd, 0, n * k * sizeof(float));
//         memset(Y, 0, batch * k * sizeof(float));
//         memcpy(cur_y, y + i, batch * sizeof(unsigned char)); 
//         memcpy(X_b, X + i * n, batch * n * sizeof(float));
//         #pragma acc update device(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 

//         matrix_dot_openacc(X_b, theta, Z, batch, n, k, block_sizes, assist_sapces); 
//         matrix_softmax_normalize_openacc(Z, batch, k);

//         vector_to_one_hot_matrix_openacc(cur_y, Y, batch, k);
        
//         matrix_minus_openacc(Z, Y, batch, k);
//         matrix_dot_trans_openacc(X_b, Z, gd, n, batch, k, block_sizes_t, assist_sapces_t); // n*100 * 100*k
//         matrix_mul_scalar_openacc(gd, batchxlr, n, k);
//         matrix_minus_openacc(theta, gd, n, k);
//     }
//     for (int i = 0; i < 3; ++i) {
//         free(assist_sapces[i]);
//         free(assist_sapces_t[i]);
//     }
//     #pragma acc exit data delete(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 
// }

// void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
// {
//     /*
//     Example function to fully train a softmax regression classifier
//     */
//     size_t size = train_data->input_dim * num_classes;
//     float *theta = new float[size];
//     memset(theta, 0, size * sizeof(float));
//     size_t size_tr = train_data->images_num * num_classes;
//     size_t size_te = test_data->images_num * num_classes;
//     float *train_result = new float[size_tr];
//     float *test_result = new float[size_te];
//     float train_loss, train_err, test_loss, test_err;
//     std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
//     std::chrono::milliseconds elapsed_time;

//     // BEGIN YOUR CODE
//     float* train_X = train_data->images_matrix;
//     unsigned char* train_y = train_data->labels_array;
//     float* test_X = test_data->images_matrix;
//     unsigned char* test_y = test_data->labels_array;
//     size_t train_X_size = train_data->images_num * train_data->input_dim;
//     size_t test_X_size = test_data->images_num * test_data->input_dim;
//     size_t train_y_size = train_data->images_num;
//     size_t test_y_size = test_data->images_num;
//     #pragma acc enter data copyin(train_X[0:train_X_size], test_X[0:test_X_size], theta[0:size],\
//         train_y[0:train_y_size], test_y[0:test_y_size],\
//         train_result[0:size_tr], test_result[0:size_te]) 

//     auto start_time = std::chrono::high_resolution_clock::now(); 
//     // assign space for matrix dot
//     const int m_train = train_data->images_num;
//     const int n_train = train_data->input_dim;
//     const int m_test = test_data->images_num;
//     const int n_test = test_data->input_dim;
//     const int k = num_classes;

//     int block_sizes_train[3];
//     int block_sizes_test[3];
//     float* assist_sapces_train[3];
//     float* assist_sapces_test[3];
//     assign_blocks(m_train, n_train, k, block_sizes_train, assist_sapces_train);
//     assign_blocks(m_test, n_test, k, block_sizes_test, assist_sapces_test);

//     for (size_t epoch = 0; epoch < epochs; epoch++)
//     {

//         memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
//         memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
//         #pragma acc update device(train_result[0:size_tr], test_result[0:size_te]) 

//         softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta,
//                                         m_train, n_train, k, lr, batch);

//         matrix_dot_openacc(train_X, theta, train_result, m_train, n_train ,k, block_sizes_train, assist_sapces_train);
//         matrix_dot_openacc(test_X, theta, test_result, m_test, n_test ,k, block_sizes_test, assist_sapces_test);
//         #pragma acc update host(theta[0:size], train_result[0:size_tr], test_result[0:size_te]) 

//         train_loss = mean_softmax_loss_openacc(train_result, train_y, train_data->images_num, num_classes);
//         test_loss = mean_softmax_loss_openacc(test_result, test_y, test_data->images_num, num_classes);
//         train_err = mean_err_openacc(train_result, train_y, train_data->images_num, num_classes);
//         test_err = mean_err_openacc(test_result, test_y, test_data->images_num, num_classes);
//         std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
//                   << std::fixed << std::setprecision(5) << train_loss << " |   "
//                   << std::fixed << std::setprecision(5) << train_err << " |   "
//                   << std::fixed << std::setprecision(5) << test_loss << " |  "
//                   << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     elapsed_time =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
//                                                               start_time);
//     std::cout << "Execution Time: " << elapsed_time.count()
//               << " milliseconds\n";
  
//    // END YOUR CODE
//     delete[] theta;
//     delete[] train_result;
//     delete[] test_result;
//     #pragma acc exit data delete(theta[0:size], train_result[0:size_tr], test_result[0:size_te]) 
// }

// float mean_softmax_loss_openacc(const float *__restrict result, const unsigned char *__restrict labels_array, size_t images_num, size_t num_classes)
// {
//     // BEGIN YOUR CODE
//     float res = 0.0f;
//     #pragma acc loop independent
//     for (int i = 0; i < images_num; ++i) {
//         int row_loc = i * num_classes;
//         float row_correct = result[row_loc + labels_array[i]];
//         float row_exp_sum = 0.0f;
//         for (int j = 0; j < num_classes; ++j) {
//             row_exp_sum += exp(result[row_loc + j]);
//         }
//         res += -row_correct + log(row_exp_sum);
//     }
//     return res / images_num;
//     // END YOUR CODE
// }

// float mean_err_openacc(const float *__restrict result, const unsigned char *__restrict labels_array, size_t images_num, size_t num_classes)
// {
//     // BEGIN YOUR CODE
//     float res = 0.0f;
//     #pragma acc loop independent
//     for (int i = 0; i < images_num; ++i) {
//         int row_idx = i * num_classes;
//         unsigned char row_max_idx = 0;
//         float row_max = result[row_idx];
//         for (int j = 1; j < num_classes; ++j) {
//             float cur = result[row_idx + j];
//             if (cur > row_max) {
//                 row_max = cur;
//                 row_max_idx = j;
//             }
//         }
//         res += (row_max_idx == labels_array[i]) ? (0) : (1);
//     }
//     return res / images_num;
//     // END YOUR CODE
// }

// void matrix_mul_openacc(float *__restrict A, const float *__restrict B, size_t size)
// {
//     // BEGIN YOUR CODE
//     #pragma acc loop independent
//     while (size > 0) {
//         (*A) *= (*B);
//         ++A;
//         ++B;
//         --size;
//     }
//     // END YOUR CODE
// }

// void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
// {
//     // BEGIN YOUR CODE

//     // END YOUR CODE
// }

// void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
// {
//     size_t size_w1 = train_data->input_dim * hidden_dim;
//     size_t size_w2 = hidden_dim * num_classes;
//     float *W1 = new float[size_w1];
//     float *W2 = new float[size_w2];
//     std::mt19937 rng;
//     rng.seed(0);
//     std::normal_distribution<float> dist(0.0, 1.0);
//     for (size_t i = 0; i < size_w1; i++)
//     {
//         W1[i] = dist(rng);
//     }
//     for (size_t i = 0; i < size_w2; i++)
//     {
//         W2[i] = dist(rng);
//     }
//     matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
//     matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
//     size_t size_tr = train_data->images_num * num_classes;
//     size_t size_te = test_data->images_num * num_classes;
//     float *train_result = new float[size_tr];
//     float *test_result = new float[size_te];
//     float train_loss, train_err, test_loss, test_err;
//     std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
//     std::chrono::milliseconds elapsed_time;
//     // BEGIN YOUR CODE
  
//     auto start_time = std::chrono::high_resolution_clock::now();
//     for (size_t epoch = 0; epoch < epochs; epoch++)
//     {
//         train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
//         train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
//         std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
//                   << std::fixed << std::setprecision(5) << train_loss << " |   "
//                   << std::fixed << std::setprecision(5) << train_err << " |   "
//                   << std::fixed << std::setprecision(5) << test_loss << " |  "
//                   << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     elapsed_time =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
//                                                               start_time);
//     std::cout << "Execution Time: " << elapsed_time.count()
//               << " milliseconds\n";
  
//     // END YOUR CODE
//     delete[] W1;
//     delete[] W2;
//     delete[] train_result;
//     delete[] test_result;
// }
