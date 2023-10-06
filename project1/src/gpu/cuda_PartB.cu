#include <iostream>
#include <vector>
#include <cuda_runtime.h> // CUDA Header

#include "utils.hpp"

// CUDA kernel functon：RGB to Gray

// __device__ void rbgarray_filtering (
//     unsigned char* r_array,
//     unsigned char* g_array,
//     unsigned char* b_array,
//     int input_jpeg_width,
//     int input_jpeg_num_channels,
//     unsigned char* input_buffer,
//     int loc,
//     float* filter,
//     int filter_offset
// ) {
//     for (int width = 1; width < input_jpeg_width - 1; ++width) {
//         r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
//         g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
//         b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset]);
//         r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);
//         g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);
//         b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 1]);          
//         r_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
//         g_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
//         b_array[width] += (unsigned char)(input_buffer[loc++] * filter[filter_offset + 2]);
//         loc -= 2 * input_jpeg_num_channels;            
//     }
// }

// attention, kernel function can't pass reference!!!
__global__ void rgbRoutine(
    unsigned char* input_buffer,
    unsigned char* output_buffer,
    int input_jpeg_width,
    int input_jpeg_num_channels,
    int input_jpeg_height,
    float* filter
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_sum[] = {0.0f, 0.0f, 0.0f};
    if ((row >= input_jpeg_height) || (row == 0) || (col == 0) || (col >= input_jpeg_width)) {
        return;
    }
    float* row_filter = filter;
    for (int input_row = row - 1; input_row < row + 2; ++input_row, row_filter+=2) {
        // for (int input_col = col - 1; input_col < col + 2; ++input_col) {
        int input_col = col - 1;
        const int rloc = (input_row * input_jpeg_width + input_col) * input_jpeg_num_channels;
        const int rloc1 = rloc + input_jpeg_num_channels;
        const int rloc2 = rloc1 + input_jpeg_num_channels;
        temp_sum[0] +=   input_buffer[rloc] * row_filter[0];
        temp_sum[1] +=   input_buffer[rloc + 1] * row_filter[0];
        temp_sum[2] +=   input_buffer[rloc + 2] * row_filter[0];
        temp_sum[0] +=  input_buffer[rloc1] * row_filter[1];
        temp_sum[1] +=  input_buffer[rloc1 + 1] * row_filter[1];
        temp_sum[2] +=  input_buffer[rloc1 + 2] * row_filter[1];
        temp_sum[0] +=  input_buffer[rloc2] * row_filter[2];
        temp_sum[1] +=  input_buffer[rloc2 + 1] * row_filter[2];
        temp_sum[2] +=  input_buffer[rloc2 + 2] * row_filter[2];
        // } 
    }
    const int output_loc = (row * input_jpeg_width + col) * input_jpeg_num_channels;
    output_buffer[output_loc]       = static_cast<unsigned char>(temp_sum[0]);
    output_buffer[output_loc + 1]   = static_cast<unsigned char>(temp_sum[1]);
    output_buffer[output_loc + 2]   = static_cast<unsigned char>(temp_sum[2]);
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

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // input_jpeg.height /= 5;

    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    float filter[9] = {1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9};
    float* filter_t = &(filter[0]);
    
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_filter;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels * sizeof(unsigned char));
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
    cudaMemcpy(d_input,
               input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height *
                input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, 
               filter_t, 
               9 * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaError_t error_1 = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error_1));

    // Computation: RGB to Gray
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // int blockSize = 512; 
    // int blockSize = 256;
    int blockdimx = 32; 
    int blockdimy = 4; 
    const dim3 blockShape(blockdimx, blockdimy);  
    const dim3 gridShape((input_jpeg.width + blockdimx - 1) / blockdimx, (input_jpeg.height + blockdimy - 1) / blockdimy);

    cudaEventRecord(start, 0); // GPU start time
    rgbRoutine<<<gridShape, blockShape>>>(
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