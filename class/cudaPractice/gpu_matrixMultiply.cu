/*
 * Purpose: Demonstrate matrix multiplication in
 * CPU and GPU with global memory and shared memory usage
 * Date and time: 04/09/2014 
 * Last modified: 
 * Author: Inanc Senocak
 *
 * to compile: nvcc -02 -o multiply.exe matrixMultiply.cu
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "timer.h"

#define SIZE 100
#define A_NROW SIZE
#define A_NCOL SIZE
#define B_NROW A_NCOL
#define B_NCOL SIZE
#define C_NROW A_NROW
#define C_NCOL B_NCOL

#define BLOCKSIZE 16 

void printMatrix( float *c )
{
    int i,j,idx;

    int nrow = 6; //C_NROW
    int ncol = 6; //C_NCOL
    printf("======RESULT=====\n");

    for (i=0; i<nrow; i++) {
        for (j=0; j<ncol; j++) {
            idx = j+i*ncol;
            printf("%8.2f ; ", c[idx]);
        }
        printf("\n");
    }
    printf("==================\n");
}

void matrixMultiplyCPU( float *a, float *b, float *c)
{
// this function does the following matrix multiplication c = a * b
// a(i x k); b(k x j); c(i x j)

    int i, j, k, idx;


//initialize matrices a & b

    for (i=0; i<A_NROW; i++) {
        for (k=0; k<A_NCOL; k++) {
             idx = k + i*A_NCOL;
             a[idx] = (float) idx;
             //printf("%8.2f ; ", a[idx]);
        }
        //printf("\n");
    }
    //printf("==================\n");
    for (k=0; k<B_NROW; k++) {   
        for (j=0; j<B_NCOL; j++) {
            idx = j+k*B_NCOL;
            b[idx] = (float) idx;
            //printf("%8.2f ; ", b[idx]);
        }
        //printf("\n");
    }
    //printf("==================\n");
    for (i=0; i<A_NROW; i++) {
        for (j=0; j<B_NCOL; j++) {
            float sum = 0.;            
            for (k=0; k<A_NCOL; k++) {
                float aa = a[k + i*A_NCOL];
                float bb = b[j + k*B_NCOL];
                sum += aa*bb; 
            }
         c[j + i*C_NCOL] = sum;
         }
     }
}

 __global__ void matrixMultiplyGPU_gl(float *a, float *b, float *c)
{
//Block index 

    int bx = blockIdx.x;
    int by = blockIdx.y;

//Thread index
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

//Row index of matrices a and c 
    
    int row = by * BLOCKSIZE + ty;

//Column index of matrices a and b 
    int col = bx * BLOCKSIZE + tx;

    float C_temp = 0.;

    for (int k = 0; k<A_NCOL; k++)
        C_temp += a[k + row*A_NCOL] * b[col + k*B_NCOL];

    c[col + row*C_NCOL] = C_temp;
}
 
 __global__ void matrixMultiplyGPU_sh(float *a, float *b, float *c)
{

//shared memory for submatrices
    __shared__ float a_sh[BLOCKSIZE][BLOCKSIZE];
    __shared__ float b_sh[BLOCKSIZE][BLOCKSIZE];

//Block index 

    int bx = blockIdx.x;
    int by = blockIdx.y;

//Thread index

    int tx = threadIdx.x;
    int ty = threadIdx.y;

//Row index of matrices a and c 

    int row = by * BLOCKSIZE + ty;

//Column index of matrices a and b 
    int col = bx * BLOCKSIZE + tx;

    float C_temp = 0.;

    for (int n=0; n < (A_NCOL/BLOCKSIZE); n++){ 

        a_sh[ty][tx] = a[ row*A_NCOL+(n*BLOCKSIZE   + tx) ];
        b_sh[ty][tx] = b[ (n*BLOCKSIZE + ty)*B_NCOL + col ];
        __syncthreads();
       
        for (int k = 0; k<BLOCKSIZE; k++)
            C_temp += a_sh[ty][k] * b_sh[k][tx];
        __syncthreads();
     }
        c[row*C_NCOL+col] = C_temp;
}

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    float *a_d, *b_d, *c_d;

    a = (float *)malloc(sizeof(float)*A_NROW*A_NCOL);
    cudaMalloc((void**) &a_d, sizeof(float)*A_NROW*A_NCOL);

    b = (float *)malloc(sizeof(float)*B_NROW*B_NCOL);
    cudaMalloc((void**) &b_d, sizeof(float)*B_NROW*B_NCOL);

    c = (float *)malloc(sizeof(float)*C_NROW*C_NCOL);
    cudaMalloc((void**) &c_d, sizeof(float)*C_NROW*C_NCOL);


    double start, finish, elapsedTime;
    float elapsedTime_gpu;

    cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
    cudaEventCreate(&timeStart);
    cudaEventCreate(&timeStop);
  
    GET_TIME(start); // need the timer.h

    matrixMultiplyCPU( a, b, c );

    GET_TIME(finish); // need the timer.h
   
    printMatrix(c);

    elapsedTime = finish - start;

    printf("elapsed wall time (CPU) = %5.4f seconds\n", elapsedTime);
    //printf("# of FLOP = %f \n", flop);
    //printf("GFLOPS = %.6f \n", (flop / elapsedTime)*1.e-9);

    //printf("Size = %i, steps = %i, wallTime = %ld, CPU Time = %f\n", NX, nTimeSteps, wallTime, CPUTime);

    cudaMemcpy( a_d, a, sizeof(float)*A_NROW*A_NCOL, cudaMemcpyHostToDevice );
    cudaMemcpy( b_d, b, sizeof(float)*B_NROW*B_NCOL, cudaMemcpyHostToDevice );

    cudaEventRecord(timeStart, 0);

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid((C_NCOL-BLOCKSIZE)/dimBlock.x + 1 , (C_NROW-BLOCKSIZE)/dimBlock.y + 1);
    //dim3 dimGrid(2,2);

    matrixMultiplyGPU_gl<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);
    //matrixMultiplyGPU_sh<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

    cudaEventRecord(timeStop, 0);
    cudaEventSynchronize(timeStop);

    //matrixMultiplyGPU_sh<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

    cudaEventElapsedTime(&elapsedTime_gpu, timeStart, timeStop);

    cudaEventDestroy(timeStart);
    cudaEventDestroy(timeStop);

    cudaMemcpy( c, c_d, sizeof(float)*C_NROW*C_NCOL, cudaMemcpyDeviceToHost );
    
    printMatrix(c);
    printf("elapsed wall time (GPU) = %5.2f ms\n", elapsedTime_gpu);

    free(a); 
    free(b); 
    free(c); 
    cudaFree(a_d); 
    cudaFree(b_d); 
    cudaFree(c_d);

    return (0);
}

