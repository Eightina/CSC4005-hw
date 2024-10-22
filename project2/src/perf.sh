#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src

# # Naive
# echo "Naive Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/naive_1024.data ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/naive_2048.data ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# Memory Locality
# echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_1024.data ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_2048.data ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_a0_1024.data ${CURRENT_DIR}/../build/src/locality_a0 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result_a0_1024.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_a0_2048.data ${CURRENT_DIR}/../build/src/locality_a0 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result_a0_2048.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_a1_1024.data ${CURRENT_DIR}/../build/src/locality_a1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result_a1_1024.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/locality_a1_2048.data ${CURRENT_DIR}/../build/src/locality_a1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result_a1_2048.txt

# echo ""

# # SIMD + Reordering
# echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/simd_1024.data ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/simd_2048.data ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
#   echo "Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/omp_1024_$num_cores.data ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
#   srun -n 1 --cpus-per-task $num_cores perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/omp_2048_$num_cores.data ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
#   echo ""
# done

# MPI + OpenMP + SIMD + Reordering
# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_1024_1.data ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_2048_1.data ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses  ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses  ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -o ${CURRENT_DIR}/../perf/mpi_1024_4.data  ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -o ${CURRENT_DIR}/../perf/mpi_1024_4.data  ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_1024_8.data ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_2048_8.data ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_1024_32.data ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 perf record -e cpu-cycles,cache-misses,page-faults,branch-misses -g -o ${CURRENT_DIR}/../perf/mpi_2048_32.data ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

#cuda
# echo "cuda Matrix Multiplication"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/targets/x86_64-linux/lib/
# srun -n 1 --gpus 1 ${CURRENT_DIR}/../build/src/gpu/cuda ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./cuda_1024.qdrep ${CURRENT_DIR}/../build/src/gpu/cuda ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt -o ./cdua_2048.qdrep ${CURRENT_DIR}/../build/src/gpu/cuda ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --gpus 1 ${CURRENT_DIR}/../build/src/gpu/cuda ${CURRENT_DIR}/../matrices/matrix_h0.txt ${CURRENT_DIR}/../matrices/matrix_h1.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --gpus 1 ${CURRENT_DIR}/../build/src/gpu/cuda ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
# echo ""