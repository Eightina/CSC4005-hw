#include "simple_ml_ext.hpp"

int main(int argc, char* argv[])
{
    std::string path_train_data_x(argv[1]);
    std::string path_train_data_y(argv[2]);
    std::string path_test_data_x(argv[3]);
    std::string path_test_data_y(argv[4]);

    DataSet* train_data = parse_mnist(path_train_data_x,
                                      path_train_data_y);
    DataSet* test_data = parse_mnist(path_test_data_x,
                                     path_test_data_y);

    // int test_m = 2, test_n=3, test_k=4;                                  
    // float test_A[6] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f};
    // float test_B[12] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f};
    // float test_C[8] = {};
    // int block_sizes[6];
    // float* assist_spaces[6];
    // assign_blocks(test_m, test_n, test_k, block_sizes, assist_spaces);
    // matrix_dot(test_A, test_B, test_C, test_m, test_n, test_k, block_sizes, assist_spaces);
    // print_matrix(test_C, test_m, test_k);

    // assign_blocks(test_m, test_n, test_k, block_sizes + 3, assist_spaces + 3);
    // float test_B_T[12] = {0.1f, 0.5f, 0.9f, 0.2f, 0.6f, 1.0f, 0.3f, 0.7f, 1.1f, 0.4f, 0.8f, 1.2f};
    // float test_D[8] = {};
    // matrix_trans_dot(test_A, test_B_T, test_D, test_m, test_n, test_k, block_sizes + 3, assist_spaces + 3);
    // print_matrix(test_D, test_m, test_k);

    std::cout << "Training two layer neural network w/ 400 hidden units" << std::endl;
    train_nn(train_data, test_data, 10, 400, 20, 0.2);
    // train_nn(train_data, test_data, 10, 100, 2, 0.2, 2000);

    delete train_data;
    delete test_data;

    return 0;
}
