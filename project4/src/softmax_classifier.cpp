#include "simple_ml_ext.hpp"

int main()
{
    DataSet* train_data = parse_mnist("/nfsmnt/223040076/coursecode/project4/dataset/training/train-images.idx3-ubyte",
                                      "/nfsmnt/223040076/coursecode/project4/dataset/training/train-labels.idx1-ubyte");
    DataSet* test_data = parse_mnist("/nfsmnt/223040076/coursecode/project4/dataset/testing/t10k-images.idx3-ubyte",
                                     "/nfsmnt/223040076/coursecode/project4/dataset/testing/t10k-labels.idx1-ubyte");
    // DataSet* train_data = parse_mnist("./dataset/training/train-images.idx3-ubyte",
    //                                   "./dataset/training/train-labels.idx1-ubyte");
    // DataSet* test_data = parse_mnist("./dataset/testing/t10k-images.idx3-ubyte",
    //"./dataset/testing/t10k-labels.idx1-ubyte");

    // int test_m = 2, test_n=3, test_k=4;                                  
    // float test_A[6] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f};
    // float test_B[12] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f};
    // float test_C[8] = {};
    // matrix_dot(test_A, test_B, test_C, test_m, test_n, test_k);
    // print_matrix(test_C, test_m, test_k);

    // float test_A_T[6] = {0.1f,0.4f,0.2f,0.5f,0.3f,0.6f};
    // float test_D[8] = {};
    // matrix_dot_trans(test_A_T, test_B, test_D, test_m, test_n, test_k);
    // print_matrix(test_D, test_m, test_k);

    // matrix_minus(test_C, test_D, test_m, test_k);
    // print_matrix(test_C, test_m, test_k);

    // matrix_mul_scalar(test_D, 2.0f, test_m, test_k);
    // print_matrix(test_D, test_m, test_k);

    // matrix_softmax_normalize(test_D, test_m, test_k);
    // print_matrix(test_D, test_m, test_k);
    // matrix_softmax_normalize(test_A, test_m, test_n);
    // print_matrix(test_A, test_m, test_n);

    std::cout << "Training softmax regression" << std::endl;
    train_softmax(train_data, test_data, 10, 10, 0.2);

    delete train_data;
    delete test_data;

    return 0;
}
