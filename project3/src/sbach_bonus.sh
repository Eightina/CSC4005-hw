#!/bin/bash
#SBATCH -o /nfsmnt/223040076/coursecode/project3/result/Project3_bonus.txt
#SBATCH -p Project
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Merge Sort
# Sequential
echo "Merge Sort Sequential (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 /nfsmnt/223040076/coursecode/project3/build/src/mergesort/mergesort_sequential 100000000
# srun -n 1 --cpus-per-task 1 /nfsmnt/223040076/coursecode/project3/build/src/mergesort/mergesort_sequential 10000
echo ""
# # Parallel
echo "Merge Sort Parallel (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores /nfsmnt/223040076/coursecode/project3/build/src/mergesort/mergesort_parallel $num_cores 100000000
  # srun -n 1 --cpus-per-task $num_cores /nfsmnt/223040076/coursecode/project3/build/src/mergesort/mergesort_parallel $num_cores 10000
done
echo ""

# Quick Sort
# Parallel
# echo "Quick Sort Parallel (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
#   echo "Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores /nfsmnt/223040076/coursecode/project3/build/src/quicksort/quicksort_parallel $num_cores 100000000
# done
# echo ""