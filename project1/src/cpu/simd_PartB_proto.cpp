//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
// const double filter[FILTER_SIZE][FILTER_SIZE] = {
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9}
// };
const __m256 filter0 = _mm256_set1_ps(1.0 / 9);
const __m256 filter1 = _mm256_set1_ps(1.0 / 9);
const __m256 filter2 = _mm256_set1_ps(1.0 / 9);

const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                -1, -1, -1, -1, 
                                -1, -1, -1, -1, 
                                -1, -1, -1, -1); 
// using -1 can mask elements not needed

forceinline __m256 line_filtering(const unsigned char* start, const __m256& filter, unsigned char* single) {
    __m128i row0 = _mm_loadu_si128((__m128i*)start);
    __m256i row0_ints = _mm256_cvtepu8_epi32(row0);
    __m256 row0_floats = _mm256_cvtepi32_ps(row0_ints);
    __m256 row0_filtered = _mm256_mul_ps(row0_floats, filter);
    (*single) += static_cast<unsigned char>(*(start + 8) * filter[0]);
    return row0_filtered;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    
    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    
    // Preprocessing: store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++) {
        for (int width = 1; width < input_jpeg.width - 1; width++) {
            int start0 = ((height - 1) * input_jpeg.width + (width - 1));
            int start1 = start0 + input_jpeg.width;
            int start2 = start1 + input_jpeg.width;
            
            // solve red
            auto temp_reds = new unsigned char[16];
            __m128i row0 = _mm_loadu_si128((__m128i*) (reds+start0));
            __m256i row0_ints = _mm256_cvtepu8_epi32(row0);
            __m256 row0_floats = _mm256_cvtepi32_ps(row0_ints);
            __m256 row0_filtered = _mm256_mul_ps(row0_floats, filter0);

            __m128i row1 = _mm_loadu_si128((__m128i*) (reds+start1));
            __m128i row1_ints = _mm_cvtepu8_epi32(row1);
            __m128 row1_floats = _mm_cvtepi32_ps(row1_ints);
            __m128 row1_filtered = _mm_mul_ps(row1_floats, filter1);

            __m128i row2 = _mm_loadu_si128((__m128i*) (reds+start2));
            __m128i row2_ints = _mm_cvtepu8_epi32(row2);
            __m128 row2_floats = _mm_cvtepi32_ps(row2_ints);
            __m128 row2_filtered = _mm_mul_ps(row2_floats, filter2);

            __m128 add_results_red = _mm_add_ps(row0_filtered, row1_filtered);
            add_results_red = _mm_add_ps(add_results_red, row2_filtered);
            __m128i add_results_red_32 =  _mm_cvtps_epi32(add_results_red);
            __m128i add_results_red_8 = _mm_shuffle_epi8(add_results_red_32, shuffle);
            _mm_storeu_si128((__m128i*)(temp_reds), add_results_red_8);
            unsigned char the_red = temp_reds[0] + temp_reds[1] + temp_reds[2];


            // https://www.jianshu.com/p/523f262c77b3
            // solve green
            auto temp_greens = new unsigned char[16];
            row0 = _mm_loadu_si128((__m128i*) (greens+start0));
            row0_ints = _mm_cvtepu8_epi32(row0);
            row0_floats = _mm_cvtepi32_ps(row0_ints);
            row0_filtered = _mm_mul_ps(row0_floats, filter0);

            row1 = _mm_loadu_si128((__m128i*) (greens+start1));
            row1_ints = _mm_cvtepu8_epi32(row1);
            row1_floats = _mm_cvtepi32_ps(row1_ints);
            row1_filtered = _mm_mul_ps(row1_floats, filter1);

            row2 = _mm_loadu_si128((__m128i*) (greens+start2));
            row2_ints = _mm_cvtepu8_epi32(row2);
            row2_floats = _mm_cvtepi32_ps(row2_ints);
            row2_filtered = _mm_mul_ps(row2_floats, filter2);

            __m128 add_results_green = _mm_add_ps(row0_filtered, row1_filtered);
            add_results_green = _mm_add_ps(add_results_green, row2_filtered);
            __m128i add_results_green_32 =  _mm_cvtps_epi32(add_results_green);
            __m128i add_results_green_8 = _mm_shuffle_epi8(add_results_green_32, shuffle);
            _mm_storeu_si128((__m128i*)(temp_greens), add_results_green_8);
            unsigned char the_green = temp_greens[0] + temp_greens[1] + temp_greens[2];

            // //solve blue
            auto temp_blues = new unsigned char[16];
            row0 = _mm_loadu_si128((__m128i*) (blues+start0));
            row0_ints = _mm_cvtepu8_epi32(row0);
            row0_floats = _mm_cvtepi32_ps(row0_ints);
            row0_filtered = _mm_mul_ps(row0_floats, filter0);

            row1 = _mm_loadu_si128((__m128i*) (blues+start1));
            row1_ints = _mm_cvtepu8_epi32(row1);
            row1_floats = _mm_cvtepi32_ps(row1_ints);
            row1_filtered = _mm_mul_ps(row1_floats, filter1);

            row2 = _mm_loadu_si128((__m128i*) (blues+start2));
            row2_ints = _mm_cvtepu8_epi32(row2);
            row2_floats = _mm_cvtepi32_ps(row2_ints);
            row2_filtered = _mm_mul_ps(row2_floats, filter2);

            __m128 add_results_blue = _mm_add_ps(row0_filtered, row1_filtered);
            add_results_blue = _mm_add_ps(add_results_blue, row2_filtered);
            __m128i add_results_blue_32 =  _mm_cvtps_epi32(add_results_blue);
            __m128i add_results_blue_8 = _mm_shuffle_epi8(add_results_blue_32, shuffle);
            _mm_storeu_si128((__m128i*)(temp_blues), add_results_blue_8);
            unsigned char the_blue = temp_blues[0] + temp_blues[1] + temp_blues[2];

            int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            filteredImage[insert_loc] = the_red;
            filteredImage[insert_loc + 1] = the_green;
            filteredImage[insert_loc + 2] = the_blue;
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