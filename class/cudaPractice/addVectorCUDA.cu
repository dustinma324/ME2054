/*
Program: addVector

A basic C/CUDA code to get started with GPU computing
The program check for GPUs on the host, prints some of the specifications
of the GPU, and then set a GPU with certain specs for computation. The CUDA
kernel illustrates allocating memory on the device, copying data to the device
and then doing a simple addition on the GPU and copying the results back to the
host and finally freeing the memory on the device.

Author: Inanc Senocak

to compile: nvcc -O2 addVectorCUDA.cu -o run.exe
to execute: ./run.exe

Demonstrate stack vs. heap memory by making NX small and large values 

*/

#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define NX 1000 //make NX a large number to test stack vs. heap
#define NY 32

__global__ void myKernel(int *a, int *b, int *c) {

  // int tid = blockIdx.x;
  // int tid = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < NX)

    a[tid] = b[tid] + c[tid];
}

void addVectors(int *a, int *b, int *c) {

  for (int i = 0; i < NX; i++) {
    a[i] = b[i] + c[i];
  }
}

int main(void) {

  // let's see how many CUDA capable GPUs we have

  int gpuCount;

  cudaGetDeviceCount(&gpuCount);

  printf(" Number of GPUs = %d\n", gpuCount);

  int myDevice = 0;

  cudaSetDevice(myDevice);

  // let's use the device to do some calculations
//  int a[NX],b[NX],c[NX];
/*
  int *a = malloc(NX * sizeof(*a));
  int *b = malloc(NX * sizeof(*b));
  int *c = malloc(NX * sizeof(*c));
*/

  int *a = (int *) malloc(NX * sizeof(*a));
  int *b = (int *) malloc(NX * sizeof(*b));
  int *c = (int *) malloc(NX * sizeof(*c));

  int *d_a, *d_b, *d_c; // create pointers for the device

  cudaMalloc((void **)&d_a, NX * sizeof(int)); // Be careful not to
                                // dereference this pointer,
                                // attach d_ to varibles
  cudaMalloc((void **)&d_b, NX * sizeof(int));
  cudaMalloc((void **)&d_c, NX * sizeof(int));

  // Let's fill the arrays with some numbers

  for (int i = 0; i < NX; i++) {

    a[i] = 0;
    b[i] = 4;
    c[i] = 1;
  }

  // Let's create the infrastructure to time the host & device operations

  double start, finish; // for the CPU

  cudaEvent_t timeStart,
      timeStop; // WARNING!!! use events only to time the device
  cudaEventCreate(&timeStart);
  cudaEventCreate(&timeStop);
  float elapsedTime; // make sure it is of type float, precision is
                     // milliseconds (ms) !!!

  GET_TIME(start);

  // Let's do the following operation on the arrays on the host: a = b +
  // c
  addVectors(a, b, c);

  GET_TIME(finish);

  printf("elapsed wall time (host) = %.6f seconds\n", finish - start);

  // Let's print the results on the screen

  printf("b, c, a=b+c\n");

  //     for (int i=0; i<NX; i++) {
  //         printf("%d %2d %3d\n", b[i], c[i], a[i]);
  //     }

  cudaMemcpy(d_b, b, NX * sizeof(int),
             cudaMemcpyHostToDevice); // memcpy(dest,src,...
  cudaMemcpy(d_c, c, NX * sizeof(int),
             cudaMemcpyHostToDevice); // memcpy(dest,src,...

  cudaEventRecord(timeStart,
                  0); // don't worry for the 2nd argument zero, it is
                      // about cuda streams
  ///*
  dim3 threadsPerBlock(16,
                       16); // Best practice of having 256 threads per block
  dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

  myKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b,
                                           d_c); // Be careful with the syntax!

  //*/
  /*     int blockSize = 256;
       int nBlocks   = (NX + blockSize -1) / blockSize; //round up if n
     is not a multiple of blocksize myKernel <<<nBlocks,
     blockSize>>>(d_a, d_b, d_c);
       //myKernel<<<1, 1>>> (d_a, d_b, d_c);*/

  printf("a[100] = %4d\n", a[100]);

  cudaEventRecord(timeStop, 0);
  cudaEventSynchronize(timeStop);

  // WARNING!!! do not simply print (timeStop-timeStart)!!

  cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);

  printf("elapsed wall time (device) = %3.1f ms\n", elapsedTime);

  cudaEventDestroy(timeStart);
  cudaEventDestroy(timeStop);

  cudaMemcpy(a, d_a, NX * sizeof(int), cudaMemcpyDeviceToHost);

  //     for (int i=0; i<NX; i++) {
  //         printf("%3d\n", a[i]);
  //     }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
