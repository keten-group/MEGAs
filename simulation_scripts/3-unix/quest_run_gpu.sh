#!/bin/bash
#SBATCH -A p32625 ## account (unchanged)
#SBATCH -p gengpu ## "-p" instead of "-q"
#SBATCH -N 1 ## number of nodes
#SBATCH -n 1 ## number of cores (not sure what is best yet)
#SBATCH -t 47:58:00 ## walltime (2 dyas max right now)
#SBATCH --job-name=unix ## name of job
#SBATCH --mem=10G
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:h100:1 ## modify this line

##### These are shell commands. Note that all MSUB commands come first.

module purge all
source /home/xys3549/hoomd-4.4-gpu/bin/activate

module load python/3.8.4
module load cuda/cuda-12.1.0-openmpi-4.1.4
module load mpi/openmpi-4.1.4-gcc.11.2.0

export LD_LIBRARY_PATH=/software/gcc/11.2.0/lib64:/software/gcc/11.2.0/lib:/software/cuda/cuda-12.1.0/targets/x86_64-linux/lib/stubs:/software/cuda/cuda-12.1.0/bin/computeprof:/software/cuda/cuda-12.1.0/lib64:/software/mpi/openmpi-4.1.4-gcc-11.2.0/lib:/software/python/3.8.4/lib64:/software/python/3.8.4/lib:/software/openssl/1.1.1u-scotty/env/lib:/home/xys3549/hoomd-env/lib/python3.8/site-packages/lib:
export LD_PRELOAD=/home/xys3549/miniconda3/lib/libembree4.so.4

mpirun -np 1 /home/xys3549/hoomd-4.4-gpu/bin/python3 3-unix.py > 3-unix.out