cd build
cmake ..
make -j4
cd ..
sbatch src/perf.sh
