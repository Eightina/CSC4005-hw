cd build
cmake ..
make -j4
cd ..
# sbatch src/sbach.sh
sbatch -N 2 src/sbach.sh