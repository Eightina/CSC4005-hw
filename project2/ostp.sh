cd build
cmake ..
make -j4
cd ..
sbatch src/sbach.sh
# sbatch -N 2 src/extend64.sh
# sbatch -N 4 src/extend64.sh