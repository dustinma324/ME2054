/***********************************************************************
 * QUIZ INSTRUCTIONS
 *
 * you are given a partially completed code. Read the code line by line
 * and implement the necessary steps to compute the 1d heat conduction
 * problem given in the handout.
 *
 ***********************************************************************
 *
 * Numerical and analytical solution of the 1D heat conduction problem
 *
 * Author: enter your name
 * Date: enter today's date
 *
 * gcc -O2 -lm -std=c99 1d_heat.c -o heat_1d.exe
 *
 * to execute: ./heat_1d.exe <simulation end time (seconds)>
 *
 * nvcc -O2 quiz_2_heat_working.cu -DSINGLE=1 -o run.exe
 */

#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define LX 3.0f /* length of the domain in x-direction  */
#define NX 32   /* includes boundary points on both end */
#define DX LX / ((REAL)(NX - 1))
#define ALPHA 1.0f
#define DT 0.25f * DX *DX / ALPHA
#define BLOCK_SIZE 256

#define RESTRICT __restrict__

#ifndef SINGLE
typedef double REAL;
typedef int    INT;
#define PI M_PI
#else
typedef float REAL;
typedef int   INT;
#define PI M_PI
#endif

__global__ void solveHeat_1D (REAL *RESTRICT unew, const REAL *RESTRICT u, const REAL *RESTRICT x)
{
    INT  i = threadIdx.x + blockIdx.x * blockDim.x;
    REAL dxi = 1.f / (DX * DX);
    REAL xc, source;

    if ( i > 0 && i < NX-1 ){
        xc     = x[ i ];
        source = -(xc * xc - 4.f * xc + 2.f) * exp(-xc); // source term
        unew[ i ] = (ALPHA * (u[ i + 1 ] - 2.0f * u[ i ] + u[ i - 1 ]) * dxi + source) * DT + u[ i ];
    }
}
void exactSolution(REAL *RESTRICT uExact, const REAL *RESTRICT x)
{
    INT i;
    for (i = 0; i < NX; i++) {
        uExact[ i ] = x[ i ] * x[ i ] * exp(-x[ i ]);
    }
}

void meshGrid(REAL *RESTRICT x)
{
    INT i;
    for (i = 0; i < NX; i++) {
        x[ i ] = DX * (( REAL ) i);
    }
}

void writeOutput(const REAL *RESTRICT x, const REAL *RESTRICT uExact, const REAL *RESTRICT u)
{
    INT   i;
    FILE *output;
    output = fopen("1d_heat.dat", "w");

    for (i = 0; i < NX; i++) {
        fprintf(output, "%10f %10f %10f\n", x[ i ], uExact[ i ], u[ i ]);
    }
    fclose(output);
}

INT main(INT argc, char *argv[])
{
    if (argc < 2) {
        perror("Command-line usage: executableName <end Time (seconds)>");
        exit(1);
    }

    REAL endTime = atof(argv[ 1 ]);

    REAL *uExact, *x;
    REAL *unew, *u, *tmp;

    //  allocate heap memory here for arrays needed in the solution algorithm
    //  read the code carefully to determine those variables

    cudaMallocManaged(&unew,NX* sizeof(*unew));
    cudaMallocManaged(&u,NX* sizeof(*u));
    cudaMallocManaged(&x,NX* sizeof(*x));

    uExact =(REAL*)calloc(NX, sizeof(*uExact));

    // calculate the x coordinates of each computational point
    meshGrid(x);
    // compute the exact solution to the 1D heat conduction problem
    exactSolution(uExact, x);

    // apply boundary conditions (make sure to apply boundary conditions to both u and unew)
    u[ 0 ]         = 0.f;
    unew[ 0 ]      = 0.f;
    unew[ NX - 1 ] = uExact[ NX - 1 ];
    u[ NX - 1 ]    = uExact[ NX - 1 ];

    REAL time = 0.f;

    int nBlocks = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    REAL   elapsedTime;   // in float because it is recorded in ms

    cudaEvent_t timeStart, timeStop; // cudaEvent_t initializes variable used in event time
    cudaEventCreate(&timeStart);
    cudaEventCreate(&timeStop);
    cudaEventRecord(timeStart, 0);

    while (time < endTime) {
        // call the solveHeat_1D( ) function here with correct parameters
        // and necessary updates on the solution array
        solveHeat_1D<<<nBlocks,BLOCK_SIZE>>>(unew, u, x);
	cudaDeviceSynchronize();
	// swap pointers
        tmp  = unew;
        unew = u;
        u    = tmp;
	// incrementing the time
        time += DT;
    }

    cudaEventRecord(timeStop, 0);
    cudaEventSynchronize(timeStop);
    cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);


    // call the writeOutput( ) function here with correct parameters

    writeOutput(x, uExact, u);

    cudaFree(unew);
    cudaFree(u);
    free(uExact);
    cudaFree(x);

    return EXIT_SUCCESS;
}
