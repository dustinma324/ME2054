################################################################################
##
## To build all versions of the hello program source this script:
## $ . build-hello.sh
##
## Comment/Uncomment lines to select between different versions
##   to build.
##
#################################################################################
##
## Set environment
##   Note: You must use the same environment on the compute
##         nodes to run the applications
##
module purge
module load intel/2017.1.132   
module load intel-mpi/2017.1.132
##
#################################################################################
##
## Compile mpi c++ version
mpiicc hello-mpi.cpp -o hello-mpi
##
## Compile mpi fortran version
mpiifort hello-mpi.f90 -o fhello-mpi
##
#################################################################################
##
## Compile openmp c++ version
icc hello-openmp.cpp -qopenmp -o hello-openmp
##
## Compile openmp fortran version
ifort hello-openmp.f90 -qopenmp -o fhello-openmp
##
#################################################################################
