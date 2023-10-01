//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <memory>
#include <vector>

#include "utils.hpp"

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
        for (int width = 1; width < input_jpeg.width - 1; width++) {

            int sum_r = 0, sum_g = 0, sum_b = 0;
            int inner_height = height - 1, inner_width = width - 1;
            // row 0
            int rloc = ((inner_height) * input_jpeg.width + (inner_width)) * input_jpeg.num_channels;
            auto cur_filter = filter.begin();
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            // row 1
            rloc += (input_jpeg.width - FILTER_SIZE) * input_jpeg.num_channels;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            // row 2
            rloc += (input_jpeg.width - FILTER_SIZE) * input_jpeg.num_channels;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc++] * (*cur_filter);
            cur_filter++;
            sum_r += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_g += input_jpeg.buffer[rloc++] * (*cur_filter);
            sum_b += input_jpeg.buffer[rloc] * (*cur_filter);
            
            // for (int i = -1; i <= 1; i++) {
            //     for (int j = -1; j <= 1; j++) {
            //         int rloc = ((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels;
            //         sum_r += input_jpeg.buffer[rloc++] * filter[0];
            //         sum_g += input_jpeg.buffer[rloc++] * filter[0];
            //         sum_b += input_jpeg.buffer[rloc] * filter[0];
            //     }
            // }

            int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            filteredImage[insert_loc++] = static_cast<unsigned char>(sum_r);
            filteredImage[insert_loc++] = static_cast<unsigned char>(sum_g);
            filteredImage[insert_loc] = static_cast<unsigned char>(sum_b);

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
