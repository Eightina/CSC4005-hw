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

    // std::vector<float>& filter;
    float filter[];
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
    float filter[],
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

// routine function to smooth RGB for a portion of the image
void* rgbRoutine(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    // [start_row, end_row)
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

        for (int width = 1; width < data->input_jpeg_width - 1; ++width) {
            const int insert_loc = (height * data->input_jpeg_width + width) * data->input_jpeg_num_channels;
            data->output_buffer[insert_loc] = r_array[width];
            data->output_buffer[insert_loc + 1] = g_array[width];
            data->output_buffer[insert_loc + 2] = b_array[width];
        } 
    }

    return nullptr;
}

int main(int argc, char** argv) {
    const int FILTER_SIZE = 3;
    float filter[9] = {1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9};

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

    // split workload
    // For example, there are 11 pixels and 3 tasks, 
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_lines = input_jpeg.height;
    int lines_num_per_task = total_lines / num_threads;
    int left_lines = total_lines % num_threads;
    std::vector<int> cuts(num_threads + 1, 0);
    int divided_left_lines_num = 0;
    for (int i = 0; i < num_threads; i++) {
        if (divided_left_lines_num < left_lines) {
            cuts[i+1] = cuts[i] + lines_num_per_task + 1;
            divided_left_lines_num++;
        } else cuts[i+1] = cuts[i] + lines_num_per_task;
    }
    cuts[0] = 1; // edge of image

    // working part
    auto start_time = std::chrono::high_resolution_clock::now();

    int chunk_size = input_jpeg.height / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].input_jpeg_width = input_jpeg.width;
        thread_data[i].input_jpeg_num_channels = input_jpeg.num_channels;

        thread_data[i].output_buffer = filteredImage;
        thread_data[i].start_row = cuts[i];
        thread_data[i].end_row = cuts[i + 1];
        
        pthread_create(&threads[i], nullptr, rgbRoutine, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
