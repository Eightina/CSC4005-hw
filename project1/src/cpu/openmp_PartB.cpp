#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>    // OpenMP header
#include "utils.hpp"

int main(int argc, char** argv) {
    const int FILTER_SIZE = 3;
    std::vector<float> filter(9, 1.0/9);

    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_processes\n";
        return -1;
    }

    // set number of processes
    int num_processes = std::stoi(argv[3]); // User-specified thread count
    omp_set_num_threads(num_processes);
    
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    
    // // Separate R, G, B channels into three continuous arrays
    // auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    // for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
    //     rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
    //     gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
    //     bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    // }

    // Transforming the R, G, B channels to Gray in parallel
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(none) shared(filter, filteredImage, input_jpeg)
    for (int height = 0; height < input_jpeg.height; height++) {
        unsigned char r_array[input_jpeg.width] = {};
        unsigned char g_array[input_jpeg.width] = {};
        unsigned char b_array[input_jpeg.width] = {};

        int rloc = ((height - 1) * input_jpeg.width) * input_jpeg.num_channels;
        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[0]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[0]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[0]);
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[1]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[1]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[1]);            
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[2]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[2]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[2]);
            rloc -= 2 * input_jpeg.num_channels;            
        }

        rloc = ((height) * input_jpeg.width) * input_jpeg.num_channels;
        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[3]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[3]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[3]);
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[4]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[4]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[4]);
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[5]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[5]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[5]);
            rloc -= 2 * input_jpeg.num_channels;  
        }

        rloc = ((height + 1) * input_jpeg.width) * input_jpeg.num_channels;
        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[6]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[6]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[6]);
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[7]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[7]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[7]);
            r_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[8]);
            g_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[8]);
            b_array[width] += (unsigned char)(input_jpeg.buffer[rloc++] * filter[8]);
            rloc -= 2 * input_jpeg.num_channels;                          
        }

        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            filteredImage[insert_loc] = r_array[width];
            filteredImage[insert_loc + 1] = g_array[width];
            filteredImage[insert_loc + 2] = b_array[width];
        }
        
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
