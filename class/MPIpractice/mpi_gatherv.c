/* Demonstrate variable gather and scatter in MPI
 *
 * Author: Inanc Senocak
 * Date: 03/02/2012
 * Last edit: 04/20/2016
 *
 * to compile: mpicc -std=c99 -O3 <source.c> -o <executable>
 * to execute: mpirun -n 7 <#processes> ./<executable>
 * Make sure to run with 7 processes.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int myRank, nProcs;

    int buffer[ 7 ]                = {7, 7, 7, 7, 7, 7, 7};
    int recv_buffer[ 13 ]          = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    int receive_counts[ 4 ]        = {1, 2, 3, 4};
    int receive_displacements[ 4 ] = {0, 1, 3, 6};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if (nProcs != 7) {
        printf("Please run with 7 processes\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 666);
    }

    if (myRank != 0) {
        for (int i = 0; i < nProcs; i++) {
            buffer[ i ] = myRank;
        }
    }
    if (myRank == 0) buffer[ 0 ] = 0;

    MPI_Gatherv(buffer + myRank, myRank + 1, MPI_INT, recv_buffer, receive_counts, receive_displacements, MPI_INT, 0,
                MPI_COMM_WORLD);

    if (myRank == 0) {
        printf("test %d %d\n", buffer[ 0 ], buffer[ 1 ]);
        for (int i = 0; i < 13; i++) {
            printf("[%d]", recv_buffer[ i ]);
        }
        printf("\n");
        fflush(stdout);
    }

    MPI_Finalize( );

    return EXIT_SUCCESS;
}
