/*
Program: addVector

This is a modification of the addVectorCUDA.cu from the class folder.
The modification was done to complete number 3 on Homework #4.
Changes made to the original program inclues function calculations, location of code statments,
and adding/removing comments. This was done so to complete the assignment as well as to understand
the logic behing parallel coding using CUDA GPU.

Author: Inanc Senocak
Editor: Dustin (Ting-Hsuan) Ma

Compile: nvcc -O2 addVectorCUDA.cu -o run.exe
Execute: ./run.exe

*/

#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define NX 10000000 //make NX a large number to test stack vs. heap
#define RADIUS 3

typedef float REAL;

__global__ void GPU_stencil(REAL *a, REAL *b, REAL *c) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < NX)
    a[tid] = b[tid] + c[tid];
}

void CPU_stencil(REAL *a, REAL *b, REAL *c) {

  for (int i = 0; i < NX; i++) {
    a[i] = b[i] + c[i];
  }
}

int main(void) {

  // Checking for Cuda capable GPU
  int gpuCount;
  cudaGetDeviceCount(&gpuCount);
  printf(" Number of GPUs = %d\n", gpuCount);
  printf("\n");
  int myDevice = 0;
  cudaSetDevice(myDevice);

  // Allocating memory for CPU
  REAL *a = (REAL*) malloc(NX * sizeof(*a));
  REAL *b = (REAL*) malloc(NX * sizeof(*b));
  REAL *c = (REAL*) malloc(NX * sizeof(*c));

  // Allocating memory for GPU
  REAL *d_a, *d_b, *d_c; // create pointers for the device
  cudaMallocManaged(&d_a, NX * sizeof(REAL));
  cudaMallocManaged(&d_b, NX * sizeof(REAL));
  cudaMallocManaged(&d_c, NX * sizeof(REAL));

  // Let's fill the arrays with some numbers
  for (int i = 0; i < NX; i++) {

    a[i] = 0.0f;
    b[i] = 2.0f;
    c[i] = 1.0f;
  }

  // *********************CPU************************
  double start, finish; // time for CPU
  REAL elapsedTime;	// in float because it is recorded in ms

  GET_TIME(start);
  
  CPU_stencil(a, b, c); // calling CPU function
  
  GET_TIME(finish);

  printf("|============================CPU============================|\n"); 
  printf("a[100] = %4f, elapsed wall time (host) = %.6f seconds \n", a[100], finish-start);
  printf("\n");
  // *********************GPU************************
  int blockSize = 256;
  int nBlocks   = (NX + blockSize -1) / blockSize;	// allows n to round up

  // Copying array memory from host to device
  cudaMemcpy(d_b, b, NX * sizeof(REAL),cudaMemcpyHostToDevice); 
  cudaMemcpy(d_c, c, NX * sizeof(REAL),cudaMemcpyHostToDevice);

  cudaEvent_t timeStart,timeStop; // cudaEvent_t initializes variable used in event time 
  cudaEventCreate(&timeStart);
  cudaEventCreate(&timeStop);
  cudaEventRecord(timeStart,0);
  
  GPU_stencil<<<nBlocks, blockSize>>> (d_a, d_b, d_c);	// replaced <<<1,1>>> with current
  
  cudaEventRecord(timeStop,0);
  cudaEventSynchronize(timeStop);
  cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);

  // Copying result array from device back to memory
  cudaMemcpy(a, d_a, NX * sizeof(REAL), cudaMemcpyDeviceToHost);

  printf("|============================GPU============================|\n");
  printf("d_a[100] = %4f, elapsed wall time (device) = %3.1f ms\n", d_a[100], elapsedTime);
  
  // Removing event created for timing the calculation
  cudaEventDestroy(timeStart);
  cudaEventDestroy(timeStop);

  // Deallocating memory used for host and device
  free(a);free(b);free(c);
  cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);

  return 0;
}
