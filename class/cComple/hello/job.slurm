#!/bin/bash
#SBATCH --job-name="hello-world"
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --cpus-per-task=1
#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --time=0-00:10:00
#SBATCH --account=me2054
#SBATCH --reservation=me2054
module load intel intel-mpi

ulimit -s unlimited
ulimit -l unlimited


mpirun -n 8 ./hello-mpi
crc-job-stats.py
