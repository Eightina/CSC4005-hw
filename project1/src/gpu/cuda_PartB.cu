//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// CUDA implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <vector>
#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

// CUDA kernel functon：RGB to Gray

__device__ void rbgarray_filtering (
    unsigned char* r_array,
    unsigned char* g_array,
    unsigned char* b_array,
    int input_jpeg_width,
    int input_jpeg_num_channels,
    unsigned char* input_buffer,
    int loc,
    float* filter,
    int filter_offset
) {
    for (int width = 1; width < input_jpeg_width - 1; ++width) {
        r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
        g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
        b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
        r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);
        g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);
        b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);          
        r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
        g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
        b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
        loc -= 2 * input_jpeg_num_channels;            
    }
}

// attention, kernel function can't pass reference!!!
__global__ void rgbRoutine(
    unsigned char* input_buffer,
    unsigned char* output_buffer,
    int input_jpeg_width,
    int input_jpeg_num_channels,
    int input_jpeg_height,
    float* filter
) {
    int height = blockIdx.x * blockDim.x + threadIdx.x;
    if (height == 0 || height >= input_jpeg_height) {
        return;
    }
    // int end_row = ((start_row + num_rows) > input_jpeg.height) ? input_jpeg.height : (start_row + num_rows);
    // [start_row, end_row)
    // for (int height = start_row; height < end_row; height++) {
    unsigned char *r_array, *g_array, *b_array;
    cudaMalloc((void**)&r_array, input_jpeg_width * sizeof(unsigned char));
    cudaMalloc((void**)&g_array, input_jpeg_width * sizeof(unsigned char));
    cudaMalloc((void**)&b_array, input_jpeg_width * sizeof(unsigned char));

    int rloc = ((height - 1) * input_jpeg_width) * input_jpeg_num_channels;
    rbgarray_filtering(r_array, g_array, b_array,
                        input_jpeg_width, input_jpeg_num_channels, 
                        input_buffer, rloc, filter, 0);

    rloc = ((height) * input_jpeg_width) * input_jpeg_num_channels;
    rbgarray_filtering(r_array, g_array, b_array,
                        input_jpeg_width, input_jpeg_num_channels, 
                        input_buffer, rloc, filter, 3);

    rloc = ((height + 1) * input_jpeg_width) * input_jpeg_num_channels;
    rbgarray_filtering(r_array, g_array, b_array,
                        input_jpeg_width, input_jpeg_num_channels, 
                        input_buffer, rloc, filter, 6);

    for (int width = 1; width < input_jpeg_width - 1; ++width) {
        int insert_loc = (height * input_jpeg_width + width) * input_jpeg_num_channels;
        output_buffer[insert_loc] = r_array[width];
        output_buffer[insert_loc + 1] = g_array[width];
        output_buffer[insert_loc + 2] = b_array[width];
    }
    
    cudaFree(r_array);
    cudaFree(g_array);
    cudaFree(b_array);
    return;
}

// __global__ void rgbToGray(const unsigned char* input, unsigned char* output,
//                           int width, int height, int num_channels) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < width * height)
//     {
//         unsigned char r = input[idx * num_channels];
//         unsigned char g = input[idx * num_channels + 1];
//         unsigned char b = input[idx * num_channels + 2];
//         output[idx] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
//     }
// }

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read and preprocess input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    float filter[9] = {1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9};
    float* filter_t = &(filter[0]);
    
    // Allocate memory on device (GPU)
    unsigned char *d_reds, *d_greens, *d_blues;
    unsigned char* d_output;
    float* d_filter;
    cudaMalloc((void**)&d_reds, input_jpeg.width * input_jpeg.height);
    cudaMalloc((void**)&d_greens, input_jpeg.width * input_jpeg.height);
    cudaMalloc((void**)&d_blues, input_jpeg.width * input_jpeg.height);
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);
    cudaMalloc((void**)&d_filter, 9 * sizeof(float));

    // check and assign heap memory
    size_t cuda_heap_size = 0;
    cudaDeviceGetLimit(&cuda_heap_size, cudaLimitMallocHeapSize);
    printf("before: heap size is %d MB\n", cuda_heap_size / 1024 / 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8 * input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels * sizeof(unsigned char));
    cudaDeviceGetLimit(&cuda_heap_size, cudaLimitMallocHeapSize);
    printf("after: heap size is %d MB\n", cuda_heap_size / 1024 / 1024);
    cudaError_t error_0 = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error_0));


    // Copy input data from host to device
    cudaMemcpy(d_reds, reds,
               input_jpeg.width * input_jpeg.height,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_greens, greens,
               input_jpeg.width * input_jpeg.height,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_blues, blues,
               input_jpeg.width * input_jpeg.height,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter_t, 
               9 * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaError_t error_1 = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error_1));

    // Computation: RGB smoothing
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // int blockSize = 512; 
    // int blockSize = 256; 
    int blockSize = input_jpeg.width / 100; // 192
    // dim3 gridSize(ceil(N/256.0), 1, 1);
    // dim3 blockSize(input_jpeg.width / 100, 1, 1);
    dim3 gridSize(input_jpeg.height - 2, 10, 3); //(12993, 10, 3)
    // int blockSize = 128; 
    // int numBlocks = input_jpeg.height / blockSize.x + 1; // each thread a line
    // int rowsPerThread = input_jpeg.height / numBlocks / blockSize;

    cudaEventRecord(start, 0); // GPU start time
    rgbRoutine<<<gridSize, blockSize>>>(
        d_input,
        d_output,
        input_jpeg.width,
        input_jpeg.num_channels,
        input_jpeg.height,
        d_filter
    );
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    cudaError_t error_3 = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error_3));
    
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height *
                input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}