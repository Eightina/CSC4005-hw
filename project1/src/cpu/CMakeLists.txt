# Required packages and flags
## MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI FOUND, Version: ${MPI_CXX_VERSION}")
endif ()
## Pthread
find_package(Threads REQUIRED)
if (Threads_FOUND)
    message("Pthread FOUND")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif ()
## OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND, Version: " ${OpenMP_CXX_VERSION})
endif ()

# Add Executables
## Sequential
add_executable(sequential_PartA
        sequential_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(sequential_PartA PRIVATE -O2)
add_executable(sequential_PartB
        sequential_PartB_optimized.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(sequential_PartB PRIVATE -O2)
# add_executable(sequential_PartB_O3
#         sequential_PartB.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(sequential_PartB_O3 PRIVATE -O2)
# add_executable(sequential_PartB_O2plus0
#         sequential_PartB.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(sequential_PartB_O2plus0 PRIVATE 
#                         -O2 -finline-functions -funswitch-loops 
#                         -fpredictive-commoning -fgcse-after-reload -fsplit-paths 
#                         -ftree-loop-distribute-patterns -fvect-cost-model
#                         -ftree-partial-pre -fpeel-loops -fipa-cp-clone)
# add_executable(sequential_PartB_O2plus1
#                 sequential_PartB.cpp
#                 ../utils.cpp ../utils.hpp)
# target_compile_options(sequential_PartB_O2plus1 PRIVATE 
#                         -O2 -ftree-loop-vectorize -ftree-slp-vectorize)           

## SIMD Vectorization (AVX2)
add_executable(simd_PartA
        simd_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(simd_PartA PRIVATE  -mavx2 -O2)

add_executable(simd_PartB
        simd_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(simd_PartB PRIVATE  -mavx2 -O2)

add_executable(simd_PartB
        simd_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(simd_PartB PRIVATE -O2 -mavx2)

## MPI
add_executable(mpi_PartA
        mpi_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mpi_PartA PRIVATE -O2)
target_include_directories(mpi_PartA PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_PartA ${MPI_LIBRARIES})

add_executable(mpi_PartB
        mpi_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(mpi_PartB PRIVATE -O3)
target_include_directories(mpi_PartB PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_PartB ${MPI_LIBRARIES})

# add_executable(mpi_PartB_O2
#         mpi_PartB.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(mpi_PartB_O2 PRIVATE -O2)
# target_include_directories(mpi_PartB_O2 PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# target_link_libraries(mpi_PartB_O2 ${MPI_LIBRARIES})

# add_executable(mpi_PartB_O2plus0
#         mpi_PartB.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(mpi_PartB_O2plus0 PRIVATE -O2 
#                         -finline-functions -funswitch-loops 
#                         -fpredictive-commoning -fgcse-after-reload -fsplit-paths 
#                         -ftree-loop-distribute-patterns -fvect-cost-model
#                         -ftree-partial-pre -fpeel-loops -fipa-cp-clone)
# target_include_directories(mpi_PartB_O2plus0 PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# target_link_libraries(mpi_PartB_O2plus0 ${MPI_LIBRARIES})

# add_executable(mpi_PartB_O2plus1
#         mpi_PartB.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(mpi_PartB_O2plus1 PRIVATE -O2 -ftree-loop-vectorize -ftree-slp-vectorize)   
# target_include_directories(mpi_PartB_O2plus1 PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# target_link_libraries(mpi_PartB_O2plus1 ${MPI_LIBRARIES})

# add_executable(mpi_PartB_spmaster
# mpi_PartB_spmaster.cpp
#         ../utils.cpp ../utils.hpp)
# target_compile_options(mpi_PartB_spmaster PRIVATE -O2)
# target_include_directories(mpi_PartB_spmaster PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# target_link_libraries(mpi_PartB_spmaster ${MPI_LIBRARIES})

## Pthread
add_executable(pthread_PartA
        pthread_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(pthread_PartA PRIVATE -O2)
target_link_libraries(pthread_PartA PRIVATE pthread)

add_executable(pthread_PartB
        pthread_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(pthread_PartB PRIVATE -O2)
target_link_libraries(pthread_PartB PRIVATE pthread)

## OpenMP
add_executable(openmp_PartA
        openmp_PartA.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(openmp_PartA PRIVATE  -fopenmp -O2)
target_include_directories(openmp_PartA PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp_PartA PRIVATE ${OpenMP_CXX_LIBRARIES})

add_executable(openmp_PartB
        openmp_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(openmp_PartB PRIVATE  -fopenmp -O2)
target_include_directories(openmp_PartB PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp_PartB PRIVATE ${OpenMP_CXX_LIBRARIES})

add_executable(openmp_simd_PartB
        openmp_simd_PartB.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(openmp_simd_PartB PRIVATE -mavx2 -fopenmp -O2)
target_include_directories(openmp_simd_PartB PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(openmp_simd_PartB PRIVATE ${OpenMP_CXX_LIBRARIES})
