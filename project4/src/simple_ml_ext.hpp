#ifndef SIMPLE_ML_EXT
#define SIMPLE_ML_EXT

#include <memory.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <random>

/**
 * Contain the MNIST Data.
 * Member:
 *    images_matrix: 1D float array containing the loaded
 *        data.  The dimensionality of the data should be
 *        (num_examples x input_dim) where 'input_dim' is the full
 *        dimension of the data, e.g., since MNIST images are 28x28, it
 *        will be 784.  Values should be of type float, and the data
 *        should be normalized to have a minimum value of 0.0 and a
 *        maximum value of 1.0 (i.e., scale original values of 0 to 0.0
 *        and 255 to 1.0).
 *
 *    labels_array: 1D unsigned char array containing the
 *        labels of the examples.  Values should be of type uint8 and
 *        for MNIST will contain the values 0-9.
 */
class DataSet
{
public:
    float *images_matrix;
    unsigned char *labels_array;
    size_t images_num;
    size_t input_dim;

    DataSet(size_t images_num, size_t input_dim);
    ~DataSet();
};

uint32_t swap_endian(uint32_t val);

DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename);

void inline assign_blocks(int M, int K, int N, int* block_sizes, float** assist_sapces);

void print_matrix(const float *A, size_t m, size_t n);

void matrix_dot(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k, int* block_sizes, float** assist_spaces);

void matrix_dot_trans(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t n, size_t m, size_t k, int* block_sizes, float** assist_spaces);

void matrix_trans_dot(const float *__restrict A, const float *__restrict B, float *__restrict C, size_t m, size_t n, size_t k, int* block_sizes, float** assist_spaces);

void matrix_minus(float *__restrict A, const float *__restrict B, size_t m, size_t n);

void matrix_mul_scalar(float *__restrict C, float scalar, size_t m, size_t n);

void matrix_div_scalar(float *__restrict C, float scalar, size_t m, size_t n);

void matrix_softmax_normalize(float *__restrict C, size_t m, size_t n);

void vector_to_one_hot_matrix(const unsigned char *__restrict y, float *__restrict Y, size_t m, size_t k);

void softmax_regression_epoch_cpp(const float *__restrict X,
                                  const unsigned char *__restrict y,
                                  float *__restrict theta,
                                  size_t m,
                                  size_t n,
                                  size_t k,
                                  float lr,
                                  size_t batch);

void train_softmax(const DataSet *train_data,
                   const DataSet *test_data,
                   size_t num_classes,
                   size_t epochs = 10,
                   float lr = 0.5,
                   size_t batch = 100);

float mean_softmax_loss(const float *__restrict result,
                        const unsigned char *__restrict labels_array,
                        size_t images_num,
                        size_t num_classes);

float mean_err(float *__restrict result,
               const unsigned char *__restrict labels_array,
               size_t images_num,
               size_t num_classes);

void matrix_mul(float *__restrict A, const float *__restrict B, size_t size);

void nn_epoch_cpp(const float *__restrict X,
                  const unsigned char *__restrict y,
                  float *__restrict W1,
                  float *__restrict W2,
                  size_t m,
                  size_t n,
                  size_t l,
                  size_t k,
                  float lr,
                  size_t batch);

void train_nn(const DataSet *train_data,
              const DataSet *test_data,
              size_t num_classes,
              size_t hidden_dim = 500,
              size_t epochs = 10,
              float lr = 0.5,
              size_t batch = 100);

#endif
