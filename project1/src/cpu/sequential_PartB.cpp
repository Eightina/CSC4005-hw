#include <iostream>
#include <cmath>
#include <chrono>
#include <memory>
#include <vector>

#include "utils.hpp"

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

forceinline void rbgarray_filtering (
    unsigned char r_array[],
    unsigned char g_array[],
    unsigned char b_array[],
    JPEGMeta& input_jpeg,
    int loc,
    std::vector<float>& filter,
    int filter_offset
) {
    for (int width = 1; width < input_jpeg.width - 1; ++width) {
        r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset]);
        g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset]);
        b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset]);
        r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 1]);
        g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 1]);
        b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 1]);          
        r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        loc -= 2 * input_jpeg.num_channels;            
    }
}

const int FILTER_SIZE = 3;
std::vector<float> filter(9, 1.0/9);

int main(int argc, char** argv) {
    
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg(read_from_jpeg(input_filename));

    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++) {
        unsigned char r_array[input_jpeg.width] = {};
        unsigned char g_array[input_jpeg.width] = {};
        unsigned char b_array[input_jpeg.width] = {};

        int rloc = ((height - 1) * input_jpeg.width) * input_jpeg.num_channels;
        rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc, filter, 0);

        rloc = ((height) * input_jpeg.width) * input_jpeg.num_channels;
        rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc, filter, 3);

        rloc = ((height + 1) * input_jpeg.width) * input_jpeg.num_channels;
        rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc, filter, 6);
 

        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            const int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
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
