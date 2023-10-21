#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src

# Naive
# echo "Naive Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../build/sample_result_128.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/sample_result_1024.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/sample_result_2048.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_a0.txt ${CURRENT_DIR}/../matrices/matrix_a1.txt ${CURRENT_DIR}/../build/sample_result_ax.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_b0.txt ${CURRENT_DIR}/../matrices/matrix_b1.txt ${CURRENT_DIR}/../build/sample_result_bx.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_c0.txt ${CURRENT_DIR}/../matrices/matrix_c1.txt ${CURRENT_DIR}/../build/sample_result_cx.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_d0.txt ${CURRENT_DIR}/../matrices/matrix_d1.txt ${CURRENT_DIR}/../build/sample_result_dx.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_e0.txt ${CURRENT_DIR}/../matrices/matrix_e1.txt ${CURRENT_DIR}/../build/sample_result_ex.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_f0.txt ${CURRENT_DIR}/../matrices/matrix_f1.txt ${CURRENT_DIR}/../build/sample_result_fx.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/sample_result_gx.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_h0.txt ${CURRENT_DIR}/../matrices/matrix_h1.txt ${CURRENT_DIR}/../build/sample_result_hx.txt
# echo ""

# # Memory Locality
# echo "Memory Locality Matrix Multiplication (Optimized with -O2)"
# # srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix3.txt ${CURRENT_DIR}/../matrices/matrix4.txt ${CURRENT_DIR}/../build/result.txt
# # srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# # srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_a0.txt ${CURRENT_DIR}/../matrices/matrix_a1.txt ${CURRENT_DIR}/../build/result_a.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_b0.txt ${CURRENT_DIR}/../matrices/matrix_b1.txt ${CURRENT_DIR}/../build/result_b.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_c0.txt ${CURRENT_DIR}/../matrices/matrix_c1.txt ${CURRENT_DIR}/../build/result_c.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_d0.txt ${CURRENT_DIR}/../matrices/matrix_d1.txt ${CURRENT_DIR}/../build/result_d.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_e0.txt ${CURRENT_DIR}/../matrices/matrix_e1.txt ${CURRENT_DIR}/../build/result_e.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_f0.txt ${CURRENT_DIR}/../matrices/matrix_f1.txt ${CURRENT_DIR}/../build/result_f.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
# echo ""

# SIMD + Reordering
# echo "SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_a0.txt ${CURRENT_DIR}/../matrices/matrix_a1.txt ${CURRENT_DIR}/../build/result_a.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_b0.txt ${CURRENT_DIR}/../matrices/matrix_b1.txt ${CURRENT_DIR}/../build/result_b.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_c0.txt ${CURRENT_DIR}/../matrices/matrix_c1.txt ${CURRENT_DIR}/../build/result_c.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_d0.txt ${CURRENT_DIR}/../matrices/matrix_d1.txt ${CURRENT_DIR}/../build/result_d.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_e0.txt ${CURRENT_DIR}/../matrices/matrix_e1.txt ${CURRENT_DIR}/../build/result_e.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_f0.txt ${CURRENT_DIR}/../matrices/matrix_f1.txt ${CURRENT_DIR}/../build/result_f.txt
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
# echo ""

# # OpenMP + SIMD + Reordering
# echo "OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
  # echo "Number of cores: $num_cores"
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
#   srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../build/result.txt
  # echo ""
# done
# for num_cores in 1 32
# do
  # echo "Number of cores: $num_cores"
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_a0.txt ${CURRENT_DIR}/../matrices/matrix_a1.txt ${CURRENT_DIR}/../build/result_a.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_b0.txt ${CURRENT_DIR}/../matrices/matrix_b1.txt ${CURRENT_DIR}/../build/result_b.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_c0.txt ${CURRENT_DIR}/../matrices/matrix_c1.txt ${CURRENT_DIR}/../build/result_c.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_d0.txt ${CURRENT_DIR}/../matrices/matrix_d1.txt ${CURRENT_DIR}/../build/result_d.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_e0.txt ${CURRENT_DIR}/../matrices/matrix_e1.txt ${CURRENT_DIR}/../build/result_e.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_f0.txt ${CURRENT_DIR}/../matrices/matrix_f1.txt ${CURRENT_DIR}/../build/result_f.txt
  # srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
#   srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_h0.txt ${CURRENT_DIR}/../matrices/matrix_h1.txt ${CURRENT_DIR}/../build/result_h.txt
#   echo ""
# done

# MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 2, Number of Threads: 16"
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../build/result.txt
# echo ""

echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_a0.txt ${CURRENT_DIR}/../matrices/matrix_a1.txt ${CURRENT_DIR}/../build/result_a.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_b0.txt ${CURRENT_DIR}/../matrices/matrix_b1.txt ${CURRENT_DIR}/../build/result_b.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_c0.txt ${CURRENT_DIR}/../matrices/matrix_c1.txt ${CURRENT_DIR}/../build/result_c.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_d0.txt ${CURRENT_DIR}/../matrices/matrix_d1.txt ${CURRENT_DIR}/../build/result_d.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_e0.txt ${CURRENT_DIR}/../matrices/matrix_e1.txt ${CURRENT_DIR}/../build/result_e.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_f0.txt ${CURRENT_DIR}/../matrices/matrix_f1.txt ${CURRENT_DIR}/../build/result_f.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_g0.txt ${CURRENT_DIR}/../matrices/matrix_g1.txt ${CURRENT_DIR}/../build/result_g.txt
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_h0.txt ${CURRENT_DIR}/../matrices/matrix_h1.txt ${CURRENT_DIR}/../build/result_h.txt
echo ""