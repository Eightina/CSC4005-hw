#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t m, size_t n, size_t input_k)
{
    // BEGIN YOUR CODE
    size_t M = m, K = n, N = input_k;    
    #pragma acc data copyin(A[0:M * K], B[0:K * N], C[0:M * N]) copyout(C[0:M * N])
    // #pragma acc data present(A[0:M * K], B[0:K * N], C[0:M * N])
    // #pragma acc update device(A[0:M * K], B[0:K * N], C[0:M * N])
    {
    #pragma acc region
    // #pragma acc parallel present(A[0:M * K], B[0:K * N], C[0:M * N]) num_gangs(1024)
    {
        // # pragma acc loop independent vector(64) 
        // for (size_t k = 0; k < K; ++k) {
        //     # pragma acc loop independent vector(64) 
        //     for (size_t i = 0; i < M; ++i) {
        //         float r1 = A[i * K + k];
        //         for (size_t j = 0; j < N; ++j) {
        //             C[i * N + j] += r1 * B[k * N + j];
        //         }
        //     }
        // }
        // why only this sequence works???
        # pragma acc loop independent vector(128) 
        for (int j = 0; j < N; j ++) {

            # pragma acc loop independent vector(16) 
            for (int i = 0; i < M ; i ++) {

                float sum = 0;
                # pragma acc loop independent vector(4) 

                for (int k = 0; k < K ; k ++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }

    }
    }
    // END YOUR CODE
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t input_k)
{
    // BEGIN YOUR CODE
    size_t M = m, K = n, N = input_k;
    #pragma acc data copyin(A[0:M * K], B[0:K * N], C[0:M * N]) copyout(C[0:M * N])
    // #pragma acc data present(A[0:M * K], B[0:K * N], C[0:M * N])
    {
    #pragma acc region
    {
        // # pragma acc loop independent vector(64) 
        // for (size_t k = 0; k < K; ++k) {
        //     // # pragma acc loop independent vector(64) 
        //     for (size_t i = 0; i < M; ++i) {
        //         float r1 = A[k * M + i];
        //         for (size_t j = 0; j < N; ++j) {
        //             C[i * N + j] += r1 * B[k * N + j];
        //         }
        //     }
        // }
        # pragma acc loop independent vector(128) 
        for (int j = 0; j < N; j ++) {

            # pragma acc loop independent vector(16) 
            for (int i = 0; i < M ; i ++) {

                float sum = 0;
                # pragma acc loop independent vector(4) 
                for (int k = 0; k < K ; k ++) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    // #pragma acc update host(C[0:M * N])
    // #pragma acc exit data
    }
    }
    // END YOUR CODE
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t input_k)
{
    // BEGIN YOUR CODE
    size_t M = m, K = n, N = input_k;
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < M; ++i) {
            float r1 = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += r1 * B[j * K + k];
            }
        }
    }
    // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(A[0:m * n], B[0:m * n])
    // #pragma acc data copyin(A[0:m * n], B[0:m * n]) copyout(A[0:m * n])
    {
    // #pragma acc region
    {
        size_t range = m * n;
        #pragma acc loop independent
        while (range > 0) {
            (*A) -= (*B);
            --range;
            ++A;
            ++B;
        }
    }
    }
    // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(C[0:m * n])
    size_t range = m * n;
    #pragma acc loop independent
    while (range > 0) {
        (*C) *= scalar;
        --range;
        ++C;
    }
    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(C[0:m * n])
    scalar = 1 / scalar;
    size_t range = m * n;
    #pragma acc loop independent
    while (range > 0) {
        (*C) *= scalar;
        --range;
        ++C;
    }
    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(C[0:m * n])
    #pragma acc loop independent 
    for (size_t i = 0; i < m; ++i) {
        const size_t rowloc = n * i;
        float* curC = C + rowloc;

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
    }
    // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(y[0:m], Y[0:m * n])
    #pragma acc loop independent
    while (m > 0) {
        *(Y + (*y)) = 1;
        --m;
        ++y;
        Y += n;
    }
    // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
   // BEGIN YOUR CODE 
    const float batchxlr = lr / batch;
    unsigned char cur_y[batch];
    float X_b[batch * n];
    float Z[batch * k];
    float gd[n * k];
    float Y[batch * k];
    
    for (int i = 0; i < m; i += batch) {
        memset(Z, 0, batch * k * sizeof(float));
        memset(gd, 0, n * k * sizeof(float));
        memset(Y, 0, batch * k * sizeof(float));

        memcpy(cur_y, y + i, batch * sizeof(unsigned char)); 
        memcpy(X_b, X + i * n, batch * n * sizeof(float));
        // #pragma acc data copyin(X_b[0:batch * n], theta[0:n * k], Z[0:batch * k])
        matrix_dot_openacc(X_b, theta, Z, batch, n, k); 
        matrix_softmax_normalize_openacc(Z, batch, k);
        // #pragma acc data copyin(cur_y[0:batch], Y[0:batch * k])
        vector_to_one_hot_matrix_openacc(cur_y, Y, batch, k);
        
        matrix_minus_openacc(Z, Y, batch, k);
        // #pragma acc data copyin(gd[0:n * k])
        matrix_dot_trans_openacc(X_b, Z, gd, n, batch, k); // n*100 * 100*k
        matrix_mul_scalar_openacc(gd, batchxlr, n, k);
        matrix_minus_openacc(theta, gd, n, k);
        // #pragma acc data copyout(theta[0:n * k])
    }
    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    /*
    Example function to fully train a softmax regression classifier
    */
    // float* train_images = train_data->images_matrix;
    // float* test_images = test_data->images_matrix;
    unsigned char* train_labels = train_data->labels_array;
    unsigned char* test_labels = test_data->labels_array;
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    // float train_loss, train_err, test_loss, test_err;
    float *loss_errs = new float[4];
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE

    const int m_train = train_data->images_num;
    const int n_train = train_data->input_dim;
    const int m_test = test_data->images_num;
    const int n_test = test_data->input_dim;
    const int k = num_classes;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
        memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));

        softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta,
                                        m_train, n_train, k, lr, batch);
        // #pragma acc data copyin(train_data->images_matrix[0:m_train * n_train], theta[0:n_train * k], train_result[0:m_train * k])
        matrix_dot_openacc(train_data->images_matrix, theta, train_result, m_train, n_train ,k);

        // #pragma acc data copyin(test_data->images_matrix[0:m_test * n_test], theta[0:n_test * k], test_result[0:m_test * k])
        matrix_dot_openacc(test_data->images_matrix, theta, test_result, m_test, n_test ,k);

        // #pragma acc data copyin(train_data->labels_array[0:m_train], test_data->labels_array[0:m_test])
        mean_softmax_loss_openacc(train_result, train_labels, train_data->images_num, num_classes, loss_errs);
        mean_err_openacc(train_result, train_labels, train_data->images_num, num_classes, loss_errs + 1);
        mean_softmax_loss_openacc(test_result, test_labels, test_data->images_num, num_classes, loss_errs + 2);
        mean_err_openacc(test_result, test_labels, test_data->images_num, num_classes, loss_errs + 3);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << loss_errs[0] << " |   "
                  << std::fixed << std::setprecision(5) << loss_errs[1] << " |   "
                  << std::fixed << std::setprecision(5) << loss_errs[2] << " |  "
                  << std::fixed << std::setprecision(5) << loss_errs[3] << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

void mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes, float* loss)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(result[0:images_num * num_classes], labels_array[0:images_num])
    float res = 0.0f;
    #pragma acc loop independent
    for (int i = 0; i < images_num; ++i) {
        int row_loc = i * num_classes;
        float row_correct = result[row_loc + labels_array[i]];
        float row_exp_sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            row_exp_sum += exp(result[row_loc + j]);
        }
        res += -row_correct + log(row_exp_sum);
    }
    res /= images_num;
    (*loss) = res;
    // return ;
    // END YOUR CODE
}

void mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes, float* err)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(result[0:images_num * num_classes], labels_array[0:images_num])
    float res = 0.0f;
    #pragma acc loop independent
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
    res /= images_num;
    (*err) = res;
    // END YOUR CODE
}

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    // #pragma acc data present(A[0:size], B[0:size])
    #pragma acc loop independent
    while (size > 0) {
        (*A) *= (*B);
        ++A;
        ++B;
        --size;
    }
    // END YOUR CODE
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    // float* train_images = train_data->images_matrix;
    // float* test_images = test_data->images_matrix;
    unsigned char* train_labels = train_data->labels_array;
    unsigned char* test_labels = test_data->labels_array;
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    // float train_loss, train_err, test_loss, test_err;
    float *loss_errs = new float[4];
    for (int i = 0; i < 4; ++i) {
        loss_errs[i] = 0.0f;
    }
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        mean_softmax_loss_openacc(train_result, train_labels, train_data->images_num, num_classes, loss_errs);
        mean_err_openacc(train_result, train_labels, train_data->images_num, num_classes, loss_errs + 1);
        mean_softmax_loss_openacc(test_result, test_labels, test_data->images_num, num_classes, loss_errs + 2);
        mean_err_openacc(test_result, test_labels, test_data->images_num, num_classes, loss_errs + 3);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << loss_errs[0] << " |   "
                  << std::fixed << std::setprecision(5) << loss_errs[1] << " |   "
                  << std::fixed << std::setprecision(5) << loss_errs[2] << " |  "
                  << std::fixed << std::setprecision(5) << loss_errs[3] << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
