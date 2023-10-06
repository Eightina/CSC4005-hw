#include <iostream>
#include <chrono>
#include <vector>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: RGB to Gray
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    // unsigned char *grayImage = new unsigned char[width * height];
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    float filter[9] = {1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9, 
                        1.0/9, 1.0/9, 1.0/9};
    float* filter_t = &(filter[0]);
    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
        buffer[i] = input_jpeg.buffer[i];
    delete[] input_jpeg.buffer;

    #pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                                buffer[0 : width * height * num_channels],\
                                filter_t[0 : 9])

    #pragma acc update device(filteredImage[0 : width * height * num_channels], \
                                buffer[0 : width * height * num_channels],\
                                filter_t[0 : 9])

    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma acc parallel present(filteredImage[0 : width * height * num_channels], \
                                buffer[0 : width * height * num_channels],\
                                filter_t[0 : 9]) \
    num_gangs(1024)
    // num_gangs(1024) num_
    {
        #pragma acc loop independent
        for (int row = 1; row < height - 1; ++row) {
            #pragma acc loop independent
            for (int col = 1; col < width - 1; ++col) {
                float temp_sum[] = {0.0f, 0.0f, 0.0f};
                float* row_filter = filter_t;
                #pragma acc loop independent
                for (int input_row = row - 1; input_row < row + 2; ++input_row, row_filter+=3) {
                    const int input_col = col - 1;
                    const int rloc = (input_row * width + input_col) * num_channels;
                    const int rloc1 = rloc + num_channels;
                    const int rloc2 = rloc1 + num_channels;
                    temp_sum[0] +=  buffer[rloc] * row_filter[0];
                    temp_sum[1] +=  buffer[rloc + 1] * row_filter[0];
                    temp_sum[2] +=  buffer[rloc + 2] * row_filter[0];
                    temp_sum[0] +=  buffer[rloc1] * row_filter[1];
                    temp_sum[1] +=  buffer[rloc1 + 1] * row_filter[1];
                    temp_sum[2] +=  buffer[rloc1 + 2] * row_filter[1];
                    temp_sum[0] +=  buffer[rloc2] * row_filter[2];
                    temp_sum[1] +=  buffer[rloc2 + 1] * row_filter[2];
                    temp_sum[2] +=  buffer[rloc2 + 2] * row_filter[2];
                }
                const int output_loc = (row * width + col) * num_channels;
                filteredImage[output_loc]       = static_cast<unsigned char>(temp_sum[0]);
                filteredImage[output_loc + 1]   = static_cast<unsigned char>(temp_sum[1]);
                filteredImage[output_loc + 2]   = static_cast<unsigned char>(temp_sum[2]);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    #pragma acc update self(filteredImage[0 : width * height * num_channels], \
                            buffer[0 : width * height * num_channels],\
                            filter_t[0 : 9])

    #pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    // delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
