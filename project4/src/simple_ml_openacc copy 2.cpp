// #include "simple_ml_openacc.hpp"

// void matrix_dot_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k) 
// {
//     int M = m, K = n, N = k;
//     // #pragma acc data copyin (A[0:M * K], B[0:K * N], C[0:M * N]) copyout (C[0:M * N])
//     {
//     // #pragma acc region
//     {
//         // # pragma acc loop independent vector(32) 
//         for (int k = 0; k < K; ++k) {    
//             // # pragma acc loop independent vector(32) 
//             for (int i = 0; i < M ; ++i) {
//                 int r1 = A[k + i * K];
//                 for (int j = 0; j < N ; ++j) {
//                     C[j + i * N] += r1 * B[j + k * N];
//                 }
//             }
//         }
//     }
//     // #pragma acc exit data delete(A[0: M * K], B[0: K * N])
//     }
// }
// /**
//  * Matrix Dot Multiplication Trans Version
//  * Efficiently compute C = A.T.dot(B)
//  * Args:
//  *     A (const float*): Matrix of size n * m
//  *     B (const float*): Matrix of size n * k
//  *     C (float*): Matrix of size m * k
//  **/
// void matrix_dot_trans_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k) 
// {
//     int M = m, K = n, N = k;
//     // #pragma acc data copyin (A[0:M * K], B[0:K * N], C[0:M * N]) copyout (C[0:M * N]) 
//     {
//     // #pragma acc region
//     {
//         // # pragma acc loop independent vector(32) 
//         for (int k = 0; k < K; ++k) {    
//             // # pragma acc loop independent vector(32) 
//             for (int i = 0; i < M ; ++i) {
//                 int r1 = A[k * M + i];
//                 for (int j = 0; j < N ; ++j) {
//                     C[j + i * N] += r1 * B[j + k * N];
//                 }
//             }
//         }
//     }
//     // #pragma acc exit data delete(A[0: M * K], B[0: K * N])
//     }
// }
// /**
//  * Matrix Dot Multiplication Trans Version 2
//  * Efficiently compute C = A.dot(B.T)
//  * Args:
//  *     A (const float*): Matrix of size m * n
//  *     B (const float*): Matrix of size k * n
//  *     C (float*): Matrix of size m * k
//  **/
// void matrix_trans_dot_openacc(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k)
// {
//     // int M = m, K = n, N = k;
//     // #pragma acc data copyin (A[0:M * K], B[0:K * N], C[0:M * N]) copyout (C[0:M * N])
//     // {
//     // #pragma acc region
//     // {
//     //     # pragma acc loop independent vector(32) 
//     //     for (int k = 0; k < K; ++k) {
//     //         // float r1_row[M];
//     //         // memcpy(r1_row, A + k * M, M * sizeof(float));
//     //         # pragma acc loop independent vector(32) 
//     //         for (int i = 0; i < M ; ++i) {
//     //             // int r1 = A[k + i * K];
//     //             for (int j = 0; j < N ; ++j) {
//     //                 C[j + i * N] += A[k + i * K] * B[j * K + k];
//     //             }
//     //         }
//     //     }
//     // }
//     // #pragma acc exit data delete(A[0: M * K], B[0: K * N])
//     // }
// }

// void matrix_minus_openacc(float *__restrict A, const float *__restrict B, size_t m, size_t n)
// {
//     size_t range = m * n;
//     // #pragma acc loop independent
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
//     // #pragma acc loop independent
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
//     // #pragma acc loop independent
//     while (range > 0) {
//         (*C) *= scalar;
//         --range;
//         ++C;
//     }
//     // END YOUR CODE
// }

// void matrix_softmax_normalize_openacc(float *__restrict C, size_t m, size_t n)
// {
//     // #pragma acc loop independent
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
//     // #pragma acc loop independent
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
//     // #pragma acc enter data create(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 

//     for (int i = 0; i < m; i += batch) {
//         memset(Z, 0, batch * k * sizeof(float));
//         memset(gd, 0, n * k * sizeof(float));
//         memset(Y, 0, batch * k * sizeof(float));
//         memcpy(cur_y, y + i, batch * sizeof(unsigned char)); 
//         memcpy(X_b, X + i * n, batch * n * sizeof(float));
//         // #pragma acc update device(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 

//         matrix_dot_openacc(X_b, theta, Z, batch, n, k); 
//         matrix_softmax_normalize(Z, batch, k);
//         // printf("matrix dot done\n");
//         vector_to_one_hot_matrix(cur_y, Y, batch, k);
        
//         matrix_minus(Z, Y, batch, k);
//         matrix_dot_trans_openacc(X_b, Z, gd, n, batch, k); // n*100 * 100*k
//         matrix_mul_scalar(gd, batchxlr, n, k);
//         matrix_minus(theta, gd, n, k);
//     }
//     // #pragma acc exit data delete(cur_y[0:batch], X_b[0:batch * n], Z[0:batch * k], gd[0:n * k], Y[0:batch * k]) 
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
//     // float* train_X = train_data->images_matrix;
//     // unsigned char* train_y = train_data->labels_array;
//     // float* test_X = test_data->images_matrix;
//     // unsigned char* test_y = test_data->labels_array;
//     // size_t train_X_size = train_data->images_num * train_data->input_dim;
//     // size_t test_X_size = test_data->images_num * test_data->input_dim;
//     // size_t train_y_size = train_data->images_num;
//     // size_t test_y_size = test_data->images_num;
//     // #pragma acc enter data copyin(train_X[0:train_X_size], test_X[0:test_X_size], theta[0:size],\
//     //     train_y[0:train_y_size], test_y[0:test_y_size],\
//     //     train_result[0:size_tr], test_result[0:size_te]) 

//     auto start_time = std::chrono::high_resolution_clock::now(); 
//     // assign space for matrix dot
//     const int m_train = train_data->images_num;
//     const int n_train = train_data->input_dim;
//     const int m_test = test_data->images_num;
//     const int n_test = test_data->input_dim;
//     const int k = num_classes;

//     // int block_sizes_train[3];
//     // int block_sizes_test[3];
//     // float* assist_sapces_train[3];
//     // float* assist_sapces_test[3];
//     // assign_blocks(m_train, n_train, k, block_sizes_train, assist_sapces_train);
//     // assign_blocks(m_test, n_test, k, block_sizes_test, assist_sapces_test);

//     for (size_t epoch = 0; epoch < epochs; epoch++)
//     {

//         memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
//         memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
//         // #pragma acc update device(train_result[0:size_tr], test_result[0:size_te]) 

//         softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta,
//                                         m_train, n_train, k, lr, batch);

//         matrix_dot_openacc(train_data->images_matrix, theta, train_result, m_train, n_train, k);
//         matrix_dot_openacc(test_data->images_matrix, theta, test_result, m_test, n_test, k);
//         // #pragma acc update host(theta[0:size], train_result[0:size_tr], test_result[0:size_te]) 

// // original todo:
//         train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
//         train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
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
//     // #pragma acc exit data delete(theta[0:size], train_result[0:size_tr], test_result[0:size_te]) 
// }

// float mean_softmax_loss_openacc(const float *__restrict result, const unsigned char *__restrict labels_array, size_t images_num, size_t num_classes)
// {
//     // BEGIN YOUR CODE
//     float res = 0.0f;
//     // #pragma acc loop independent
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
//     // #pragma acc loop independent
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
//     // #pragma acc loop independent
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
