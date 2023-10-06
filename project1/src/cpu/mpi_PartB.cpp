#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif


forceinline void rbgarray_filtering (
    float g_array[],
    float b_array[],
    float r_array[],
    JPEGMeta& input_jpeg,
    int loc,
    std::vector<float>& filter,
    int filter_offset
) {
    for (int width = 1; width < input_jpeg.width - 1; ++width) {
        r_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset]);
        g_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset]);
        b_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset]);
        // ++filter_iter;
        r_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 1]);
        g_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 1]);
        b_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 1]);
        // ++filter_iter;            
        r_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        g_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        b_array[width] += (input_jpeg.buffer[loc++] * filter[filter_offset + 2]);
        loc -= 2 * input_jpeg.num_channels;            
    }
}


int main(int argc, char** argv) {

    const int FILTER_SIZE = 3;
    std::vector<float> filter(9, 1.0/9);
    
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Divide the task
    // For example, there are 11 pixels and 3 tasks, 
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_lines = input_jpeg.height;
    int lines_num_per_task = total_lines / numtasks;
    int left_lines = total_lines % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_lines_num = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_lines_num < left_lines) {
            cuts[i+1] = cuts[i] + lines_num_per_task + 1;
            divided_left_lines_num++;
        } else cuts[i+1] = cuts[i] + lines_num_per_task;
    }
    cuts[0] += 1; // edge of image
    cuts.back() -= 1;

    auto start_time = std::chrono::high_resolution_clock::now();

    // The tasks for the master executor
    // 1. Transform the first division of the RGB contents to the Gray contents
    // 2. Receive the transformed Gray contents from slave executors
    // 3. Write the Gray contents to the JPEG File
    if (taskid == MASTER) {
        // Transform the first division of RGB Contents to the gray contents
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

        float r_array[input_jpeg.width] = {};
        float g_array[input_jpeg.width] = {};
        float b_array[input_jpeg.width] = {};

        for (int row = cuts[MASTER]; row < cuts[MASTER + 1]; ++row) {

            // auto filter_iter = filter.begin();
            const int rloc0 = ((row - 1) * input_jpeg.width) * input_jpeg.num_channels;
            const int rloc1 = rloc0 + input_jpeg.width * input_jpeg.num_channels;
            const int rloc2 = rloc1 + input_jpeg.width * input_jpeg.num_channels;

            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc0, filter, 0);
            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc1, filter, 3);
            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc2, filter, 6);
            
            for (int width = 1; width < input_jpeg.width - 1; ++width) {
                const int insert_loc = (row * input_jpeg.width + width) * input_jpeg.num_channels;
                filteredImage[insert_loc] = (unsigned char)r_array[width];
                filteredImage[insert_loc + 1] = (unsigned char)g_array[width];
                filteredImage[insert_loc + 2] = (unsigned char)b_array[width];
            }            
        }

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + (cuts[i] - 2 * i) * input_jpeg.width * input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i]) * input_jpeg.width * input_jpeg.num_channels;
            MPI_Recv(start_pos, length - 2 * input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        

        // const char* output_filepath = argv[2];
        // std::cout << "Output file to: " << output_filepath << "\n";
        // JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, 1, JCS_GRAYSCALE};
        // if (write_to_jpeg(output_jpeg, output_filepath)) {
        //     std::cerr << "Failed to write output JPEG to file\n";
        //     MPI_Finalize();
        //     return -1;
        // }
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } 
    // The tasks for the slave executor
    // 1. Transform the RGB contents to the Gray contents
    // 2. Send the transformed Gray contents back to the master executor
    else {
        // Transform the RGB Contents to the gray contents
        const int rows = cuts[taskid + 1] - cuts[taskid];
        const int length = rows * input_jpeg.width * input_jpeg.num_channels; 
        auto filteredImage = new unsigned char[length];
        for (int i = 0; i < length; ++i)
            filteredImage[i] = 0;
        
        int temp_row = 1;

        float r_array[input_jpeg.width] = {};
        float g_array[input_jpeg.width] = {};
        float b_array[input_jpeg.width] = {};

        for (int row = cuts[taskid]; row < cuts[taskid + 1]; ++row) {

            // auto filter_iter = filter.begin();
            const int rloc0 = ((row - 1) * input_jpeg.width) * input_jpeg.num_channels;
            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc0, filter, 0);
            // filter_iter = filter.begin();
            const int rloc1 = ((row) * input_jpeg.width) * input_jpeg.num_channels;
            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc1, filter, 3);
            // filter_iter = filter.begin();
            const int rloc2 = ((row + 1) * input_jpeg.width) * input_jpeg.num_channels;
            rbgarray_filtering(r_array, g_array, b_array, input_jpeg, rloc2, filter, 6);

            for (int width = 1; width < input_jpeg.width - 1; ++width) {
                const int insert_loc = ((temp_row) * input_jpeg.width + width) * input_jpeg.num_channels;
                filteredImage[insert_loc] = (unsigned char)r_array[width];
                filteredImage[insert_loc + 1] = (unsigned char)g_array[width];
                filteredImage[insert_loc + 2] = (unsigned char)b_array[width];
            }
            ++temp_row;     
        }

        // Send the gray image back to the master
        MPI_Send(filteredImage + input_jpeg.width * input_jpeg.num_channels, length - 2 * input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        
        // Release the memory
        delete[] filteredImage;
        // delete[] input_jpeg.buffer;
    }

    MPI_Finalize();
    return 0;
}
