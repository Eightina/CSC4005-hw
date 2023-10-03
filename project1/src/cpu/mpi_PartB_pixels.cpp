// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <mpi.h>    // MPI Header

// #include "utils.hpp"

// #define MASTER 0
// #define TAG_GATHER 0

// #ifdef _MSC_VER_ // for MSVC
// #define forceinline __forceinline
// #elif defined __GNUC__ // for gcc on Linux/Apple OS X
// #define forceinline __inline__ __attribute__((always_inline))
// #else
// #define forceinline
// #endif


// forceinline void rbgarray_filtering (
//     unsigned char r_array[],
//     unsigned char g_array[],
//     unsigned char b_array[],
//     JPEGMeta& input_jpeg,
//     int loc,
//     std::vector<int>& filter
// ) {
//     for (int width = 1; width < input_jpeg.width - 1; ++width) {
//         r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[0]);
//         g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[0]);
//         b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[0]);
//         r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[1]);
//         g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[1]);
//         b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[1]);            
//         r_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[2]);
//         g_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[2]);
//         b_array[width] += (unsigned char)(input_jpeg.buffer[loc++] * filter[2]);
//         loc -= 2 * input_jpeg.num_channels;            
//     }
// }


// int main(int argc, char** argv) {

//     const int FILTER_SIZE = 3;
//     std::vector<float> filter(9, 1.0/9);
    
//     // Verify input argument format
//     if (argc != 3) {
//         std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
//         return -1;
//     }

//     // Start the MPI
//     MPI_Init(&argc, &argv);
//     // How many processes are running
//     int numtasks;
//     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
//     // What's my rank?
//     int taskid;
//     MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
//     // Which node am I running on?
//     int len;
//     char hostname[MPI_MAX_PROCESSOR_NAME];
//     MPI_Get_processor_name(hostname, &len);
//     MPI_Status status;

//     // Read JPEG File
//     const char * input_filepath = argv[1];
//     std::cout << "Input file from: " << input_filepath << "\n";
//     auto input_jpeg = read_from_jpeg(input_filepath);
//     if (input_jpeg.buffer == NULL) {
//         std::cerr << "Failed to read input JPEG image\n";
//         return -1;
//     }

//     auto start_time = std::chrono::high_resolution_clock::now();

//     // Divide the task
//     // For example, there are 11 pixels and 3 tasks, 
//     // we try to divide to 4 4 3 instead of 3 3 5
//     int total_pixel_num = (input_jpeg.width - 2) * (input_jpeg.height - 2);
//     int pixel_num_per_task = total_pixel_num / numtasks;
//     int left_pixel_num = total_pixel_num % numtasks;

//     std::vector<int> cuts(numtasks + 1, 0);
//     int divided_left_pixel_num = 0;

//     for (int i = 0; i < numtasks; i++) {
//         if (divided_left_pixel_num < left_pixel_num) {
//             cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
//             divided_left_pixel_num++;
//         } else cuts[i+1] = cuts[i] + pixel_num_per_task;
//     }

//     // The tasks for the master executor
//     // 1. Transform the first division of the RGB contents to the Gray contents
//     // 2. Receive the transformed Gray contents from slave executors
//     // 3. Write the Gray contents to the JPEG File
//     if (taskid == MASTER) {
//         // Transform the first division of RGB Contents to the gray contents
//         auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
//         for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
//         filteredImage[i] = 0;

//         for (int i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
//             unsigned char r = input_jpeg.buffer[i * input_jpeg.num_channels];
//             unsigned char g = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
//             unsigned char b = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
//             grayImage[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
//         }

//         // Receive the transformed Gray contents from each slave executors
//         for (int i = MASTER + 1; i < numtasks; i++) {
//             unsigned char* start_pos = grayImage + cuts[i];
//             int length = cuts[i+1] - cuts[i];
//             MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
//         }

//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        

//         // Save the Gray Image
//         const char* output_filepath = argv[2];
//         std::cout << "Output file to: " << output_filepath << "\n";
//         JPEGMeta output_jpeg{grayImage, input_jpeg.width, input_jpeg.height, 1, JCS_GRAYSCALE};
//         if (write_to_jpeg(output_jpeg, output_filepath)) {
//             std::cerr << "Failed to write output JPEG to file\n";
//             MPI_Finalize();
//             return -1;
//         }

//         // Release the memory
//         delete[] input_jpeg.buffer;
//         delete[] grayImage;
//         std::cout << "Transformation Complete!" << std::endl;
//         std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
//     } 
//     // The tasks for the slave executor
//     // 1. Transform the RGB contents to the Gray contents
//     // 2. Send the transformed Gray contents back to the master executor
//     else {
//         // Transform the RGB Contents to the gray contents
//         int length = cuts[taskid + 1] - cuts[taskid]; 
//         auto grayImage = new unsigned char[length];
//         for (int i = cuts[taskid]; i < cuts[taskid + 1]; i++) {
//             unsigned char r = input_jpeg.buffer[i * input_jpeg.num_channels];
//             unsigned char g = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
//             unsigned char b = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
//             int j = i-cuts[taskid];
//             grayImage[j] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
//         }

//         // Send the gray image back to the master
//         MPI_Send(grayImage, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        
//         // Release the memory
//         delete[] grayImage;
//     }

//     MPI_Finalize();
//     return 0;
// }
