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

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

const int FILTER_SIZE = 3;

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

forceinline void store_res(__m256 row0_filtered, __m256 row1_filtered, __m256 row2_filtered, unsigned char* temp_reds, int store_offset) {
    __m256 add_results_red = _mm256_add_ps(row0_filtered, row1_filtered);
    add_results_red = _mm256_add_ps(add_results_red, row2_filtered);
    __m256i add_results_red_32 =  _mm256_cvtps_epi32(add_results_red);
    // remind that shuffle instruction works on two splitted sections, must split 256 into 128 * 2
    __m128i low = _mm256_castsi256_si128(add_results_red_32);
    __m128i high = _mm256_extracti128_si256(add_results_red_32, 1);

    __m128i add_results_red_4low = _mm_shuffle_epi8(low, shuffle);
    __m128i add_results_red_4high = _mm_shuffle_epi8(high, shuffle);

    _mm_storeu_si128((__m128i*)(&temp_reds[store_offset]), add_results_red_4low);
    _mm_storeu_si128((__m128i*)(&temp_reds[store_offset + 4]), add_results_red_4high);
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
        for (int width = 1; width < input_jpeg.width - 4; width+=3) {
            const int distance = FILTER_SIZE * input_jpeg.num_channels;

            const int start0 = ((height - 1) * input_jpeg.width + (width - 1)) * input_jpeg.num_channels;
            const int start1 = start0 + input_jpeg.width * input_jpeg.num_channels;
            const int start2 = start1 + input_jpeg.width * input_jpeg.num_channels;

            const int start0a = start0 + distance;
            const int start1a = start1 + distance;
            const int start2a = start2 + distance;

            const int start0b = start0a + distance;
            const int start1b = start1a + distance;
            const int start2b = start2a + distance;

            auto temp_reds = new unsigned char[8];
            auto temp_redsa = new unsigned char[8];
            auto temp_redsb = new unsigned char[8];

            unsigned char single = 0, singlea = 0, singleb = 0;

            __m256 row0_filtered = line_filtering((input_jpeg.buffer + start0), filter0, &single);
            __m256 row0a_filtered = line_filtering((input_jpeg.buffer + start0a), filter0, &singlea);
            __m256 row0b_filtered = line_filtering((input_jpeg.buffer + start0b), filter0, &singleb);

            __m256 row1_filtered = line_filtering((input_jpeg.buffer + start1), filter1, &single);
            __m256 row1a_filtered = line_filtering((input_jpeg.buffer + start1a), filter1, &singlea);
            __m256 row1b_filtered = line_filtering((input_jpeg.buffer + start1b), filter1, &singleb);
            
            __m256 row2_filtered = line_filtering((input_jpeg.buffer + start2), filter2, &single);
            __m256 row2a_filtered = line_filtering((input_jpeg.buffer + start2a), filter2, &singlea);
            __m256 row2b_filtered = line_filtering((input_jpeg.buffer + start2b), filter2, &singleb);
        
            // process result
            store_res(row0_filtered, row1_filtered, row2_filtered, temp_reds, 0);
            store_res(row0a_filtered, row1a_filtered, row2a_filtered, temp_redsa, 0);
            store_res(row0b_filtered, row1b_filtered, row2b_filtered, temp_redsb, 0);

            const int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            filteredImage[insert_loc]       = temp_reds[0] + temp_reds[3] + temp_reds[6];
            filteredImage[insert_loc + 1]   = temp_reds[1] + temp_reds[4] + temp_reds[7];
            filteredImage[insert_loc + 2]   = temp_reds[2] + temp_reds[5] + single;

            filteredImage[insert_loc + 3]   = temp_redsa[0] + temp_redsa[3] + temp_redsa[6];
            filteredImage[insert_loc + 4]   = temp_redsa[1] + temp_redsa[4] + temp_redsa[7];
            filteredImage[insert_loc + 5]   = temp_redsa[2] + temp_redsa[5] + singlea;

            filteredImage[insert_loc + 6]   = temp_redsb[0] + temp_redsb[3] + temp_redsb[6];
            filteredImage[insert_loc + 7]   = temp_redsb[1] + temp_redsb[4] + temp_redsb[7];
            filteredImage[insert_loc + 8]   = temp_redsb[2] + temp_redsb[5] + singleb;

            delete[] temp_reds;
            delete[] temp_redsa;
            delete[] temp_redsb;
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