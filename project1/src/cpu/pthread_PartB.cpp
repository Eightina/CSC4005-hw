#include <iostream>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"
#include <vector>


// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int start_row;
    int end_row;

    int input_jpeg_width;

};

// Function to convert RGB to Grayscale for a portion of the image
void* rgbToGray(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    
    for (int height = data->start_row; height < data->end_row; height++) {
        unsigned char r_array[data->input_jpeg_width] = {};
        unsigned char g_array[data->input_jpeg_width] = {};
        unsigned char b_array[data->input_jpeg_width] = {};


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

    // Computation: RGB to Gray
    auto grayImage = new unsigned char[input_jpeg.width * input_jpeg.height];
    
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
