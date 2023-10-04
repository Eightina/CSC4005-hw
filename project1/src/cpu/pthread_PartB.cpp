#include <iostream>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"
#include <vector>

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    int input_jpeg_width;
    int input_jpeg_num_channels;

    unsigned char* output_buffer;
    int start_row;
    int end_row;

    std::vector<float>& filter;
};

// filter func
forceinline void rbgarray_filtering (
    unsigned char r_array[],
    unsigned char g_array[],
    unsigned char b_array[],
    // JPEGMeta& input_jpeg,
    int input_jpeg_width,
    int input_jpeg_num_channels,
    unsigned char* input_buffer,
    int loc,
    std::vector<float>& filter,
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

// routine function to convert RGB to Grayscale for a portion of the image
void* rgbToGray(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    
    for (int height = data->start_row; height < data->end_row; height++) {
        unsigned char r_array[data->input_jpeg_width] = {};
        unsigned char g_array[data->input_jpeg_width] = {};
        unsigned char b_array[data->input_jpeg_width] = {};

        int rloc = ((height - 1) * data->input_jpeg_width) * data->input_jpeg_num_channels;
        rbgarray_filtering(r_array, g_array, b_array,
                            data->input_jpeg_width, data->input_jpeg_num_channels, data->input_buffer,
                            rloc, data->filter, 0);

        rloc = ((height) * data->input_jpeg_width) * data->input_jpeg_num_channels;
        rbgarray_filtering(r_array, g_array, b_array,
                            data->input_jpeg_width, data->input_jpeg_num_channels, data->input_buffer,
                            rloc, data->filter, 3);

        rloc = ((height + 1) * data->input_jpeg_width) * data->input_jpeg_num_channels;
        rbgarray_filtering(r_array, g_array, b_array,
                            data->input_jpeg_width, data->input_jpeg_num_channels, data->input_buffer,
                            rloc, data->filter, 6);
    }

    return nullptr;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = input_jpeg.width * input_jpeg.height / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = grayImage;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? input_jpeg.width * input_jpeg.height : (i + 1) * chunk_size;
        
        pthread_create(&threads[i], nullptr, rgbToGray, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1, JCS_GRAYSCALE};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] grayImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
