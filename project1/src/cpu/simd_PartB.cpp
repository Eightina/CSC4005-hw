#include <iostream>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <emmintrin.h>
#include "utils.hpp"

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

const int FILTER_SIZE = 3;
// const double filter[FILTER_SIZE][FILTER_SIZE] = {
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9}
// };
const __m128 filter0 = _mm_set1_ps(1.0 / 9);
const __m128 filter1 = _mm_set1_ps(1.0 / 9);
const __m128 filter2 = _mm_set1_ps(1.0 / 9);

const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                -1, -1, -1, -1, 
                                -1, -1, -1, -1, 
                                -1, -1, -1, -1); 
// using -1 can mask elements not needed

forceinline void line_filtering_genline(
    unsigned char* src,
    const __m128& filter,
    unsigned char* tar
) {
    __m128i row = _mm_loadu_si128((__m128i*)src);
    __m128i row_ints = _mm_cvtepu8_epi32(row);
    __m128 row_floats = _mm_cvtepi32_ps(row_ints);
    __m128 row_filtered = _mm_mul_ps(row_floats, filter);
    __m128i row_filtered_32int =  _mm_cvtps_epi32(row_filtered);

    __m128i pixel_a_u8 = _mm_shuffle_epi8(row_filtered_32int, shuffle);
    _mm_storeu_si128((__m128i*)(tar), pixel_a_u8);
    // tar[3] = 0; //padding here is not essential
}

forceinline void line_filtering_addin(
    const unsigned char* src,
    const __m128& filter,
    unsigned char* tar
) {
    __m128i row = _mm_loadu_si128((__m128i*)src);
    __m128i row_ints = _mm_cvtepu8_epi32(row);
    __m128 row_floats = _mm_cvtepi32_ps(row_ints);
    __m128 row_filtered = _mm_mul_ps(row_floats, filter);
    __m128i row_filtered_32int = _mm_cvtps_epi32(row_filtered);
    
    // fetch previous calculate result
    __m128i pre_row = _mm_loadu_si128((__m128i*)tar);
    __m128i pre_row_32int = _mm_cvtepu8_epi32(pre_row);
    row_filtered_32int = _mm_add_epi32(row_filtered_32int, pre_row_32int);

    __m128i pixel_a_u8 = _mm_shuffle_epi8(row_filtered_32int, shuffle);

    // _mm_storeu_si128((__m128i*)(tar), pixel_a_u8);  !!!wrong!!! overwriting following data in *tar
    // _mm_storeu_si64((__m128i*)(tar), pixel_a_u8);    no supported sse intrinsic
    // for efficient storing back, meanwhile not overweriting following data
    // _mm_storeu_si64() is not supported, somehow, so some pointer trick is appllied here
    // which achieves similar performance to sse instruction _mm_storeu_si64(),
    // guess that complier does similar jobs for both
    u_int32_t* temp_p = reinterpret_cast<u_int32_t*>(&pixel_a_u8);
    u_int32_t* temp_tar = reinterpret_cast<u_int32_t*>(tar);
    // unsigned char* temp_p = (u_int32_t*)&pixel_a_u8;
    *temp_tar = *temp_p;
    // tar[3] = 0; //padding here is not essential
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
    
    // Preprocess, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int height = 1; height < input_jpeg.height - 1; height++) {
        const int array_len = input_jpeg.width * 4 + 16;  // each pixel needs 4, padding, plus 16 in case of overflow
        unsigned char r_array[array_len] = {};
        unsigned char g_array[array_len] = {};
        unsigned char b_array[array_len] = {};
        const int start = ((height - 1) * input_jpeg.width);
        const int start0 = ((height) * input_jpeg.width);
        const int start1 = ((height + 1) * input_jpeg.width);

        // row 0 init filtering 三行合在一个循化里比拆开来快多了(当然，比九行合在一起也要快多了，顺序访问的重要性)
        int cbuffer_loc = start;
        for (int width = 1; width < input_jpeg.width - 1; width++) {
            line_filtering_genline(&(reds[cbuffer_loc]), filter0, &(r_array[width * 4]));    
            line_filtering_genline(&(greens[cbuffer_loc]), filter0, &(g_array[width * 4]));    
            line_filtering_genline(&(blues[cbuffer_loc++]), filter0, &(b_array[width * 4]));    
        }

        // row 1 add up to previous result
        cbuffer_loc = start0;
        for (int width = 1; width < input_jpeg.width - 1; width++) {
            line_filtering_addin(&(reds[cbuffer_loc]), filter1, &(r_array[width * 4]));    
            line_filtering_addin(&(greens[cbuffer_loc]), filter1, &(g_array[width * 4]));    
            line_filtering_addin(&(blues[cbuffer_loc++]), filter1, &(b_array[width * 4]));    
        }

        // row 2 add up to previous result
        cbuffer_loc = start1;
        for (int width = 1; width < input_jpeg.width - 1; width++) {
            line_filtering_addin(&(reds[cbuffer_loc]), filter2, &(r_array[width * 4]));    
            line_filtering_addin(&(greens[cbuffer_loc]), filter2, &(g_array[width * 4]));    
            line_filtering_addin(&(blues[cbuffer_loc++]), filter2, &(b_array[width * 4]));    
        }

        for (int width = 1; width < input_jpeg.width - 1; ++width) {
            const int insert_loc = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            const int array_loc = width * 4;
            filteredImage[insert_loc] = r_array[array_loc] + r_array[array_loc + 1] + r_array[array_loc + 2];
            filteredImage[insert_loc + 1] = g_array[array_loc] + g_array[array_loc + 1] + g_array[array_loc + 2];
            filteredImage[insert_loc + 2] = b_array[array_loc] + b_array[array_loc + 1] + b_array[array_loc + 2];
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