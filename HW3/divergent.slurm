#!/bin/bash
#SBATCH --job-name=divergentDouble
#SBATCH --output=divergentDouble.o%j
#SBATCH --error=divergentDouble.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --time=1:00:00

./divergentDouble.exe
