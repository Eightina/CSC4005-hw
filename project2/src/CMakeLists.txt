include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Required packages and flags
## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()
## OpenMPmake
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

# Add Executables
## Naive
add_executable(naive
        naive.cpp
        matrix.cpp matrix.hpp)
target_compile_options(naive PRIVATE -O2)

## Memory Locality
add_executable(locality
        locality.cpp
        matrix.cpp matrix.hpp)
target_compile_options(locality PRIVATE -O2 -fno-strict-aliasing)
# target_compile_options(locality PRIVATE -O0 -fno-strict-aliasing)
# target_compile_options(locality PRIVATE -O2 -ftree-loop-vectorize -ftree-slp-vectorize -fno-strict-aliasing)
# target_compile_options(locality PRIVATE -O3)
add_executable(locality_a0
locality_a0.cpp
        matrix.cpp matrix.hpp)
target_compile_options(locality_a0 PRIVATE -O2 -fno-strict-aliasing)
add_executable(locality_a1
locality_a1.cpp
        matrix.cpp matrix.hpp)
target_compile_options(locality_a1 PRIVATE -O2 -fno-strict-aliasing)

## SIMD Vectorization (AVX2)
add_executable(simd
        simd.cpp
        matrix.cpp matrix.hpp)
# target_compile_options(simd PRIVATE -O0 -mavx2 -mavx512f -march=native)
# target_compile_options(simd PRIVATE -O2 -mavx2)
target_compile_options(simd PRIVATE -O2 -mavx512f -fno-strict-aliasing -march=native)
# target_compile_options(simd PRIVATE -O2 -mavx512f)
# target_compile_options(simd PRIVATE -O2 -mavx512f -ftree-loop-vectorize -ftree-slp-vectorize -fno-strict-aliasing -ffast-math)
# target_compile_options(simd PRIVATE -O3 -mavx512f)

## OpenMP
add_executable(openmp
        openmp.cpp
        matrix.cpp matrix.hpp)
target_compile_options(openmp PRIVATE -O2 -mavx512f -fno-strict-aliasing -fopenmp -march=native)
# target_compile_options(openmp PRIVATE -g3 -O0 -mavx512f -fno-strict-aliasing -fopenmp)
# target_compile_options(openmp PRIVATE -O2 -fopenmp -mavx512f -ftree-loop-vectorize -ftree-slp-vectorize -fno-strict-aliasing -ffast-math)
# target_compile_options(openmp PRIVATE -O3 -fopenmp -mavx512f -fno-strict-aliasing)
target_include_directories(openmp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp PRIVATE ${OpenMP_CXX_LIBRARIES})

## MPI
add_executable(mpi
        mpi.cpp
        matrix.cpp matrix.hpp)
# target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx2)
# target_compile_options(mpi PRIVATE -O0 -g3 -fopenmp -mavx512f)
target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx512f -fno-strict-aliasing -march=native)
# target_compile_options(mpi PRIVATE -O2 -fopenmp -mavx512f -ftree-loop-vectorize -ftree-slp-vectorize -fno-strict-aliasing -ffast-math)
target_include_directories(mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mpi ${MPI_LIBRARIES} ${OpenMP_CXX_LIBRARIES})


add_subdirectory(gpu)
