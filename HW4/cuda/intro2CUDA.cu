/*
Program: intro2CUDA

A basic C/CUDA code to get started with GPU computing
The program check for GPUs on the host, prints some of the specifications
of the GPU, and then set a GPU with certain specs for computation. The CUDA
kernel
illustrates allocating memory on the device, copying data to the device and then
doing
a simple addition on the GPU and copying the results back to the host and
finally freeing the
memory on the device.

Author: Inanc Senocak

to compile: nvcc -O2 -gencode arch=compute_61,code=sm_61 intro2CUDA.cu -o run.exe
to execute: ./run.exe

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>

__global__ void myKernel(int a, int b, int *c) { *c = a + b; }

int main(void) {

  // let's see how many CUDA capable GPUs we have

  int gpuCount;

  cudaGetDeviceCount(&gpuCount);

  printf(" Number of GPUs = %d\n", gpuCount);

  // let's check the device properties

  cudaDeviceProp gpuSpecs;

  for (int i = 0; i < gpuCount; i++) {

    cudaGetDeviceProperties(&gpuSpecs, i);

    printf("GPU Name: %s\n", gpuSpecs.name);
    printf("Total Global Memory: %ld\n", gpuSpecs.totalGlobalMem);
    printf("Compute Capability: %d.%d\n", gpuSpecs.major, gpuSpecs.minor);
  }

  // let's make sure that we use the device that we want. There can be multiple
  // GPUs on a computer

  //gpuSpecs.major = 3;
  //gpuSpecs.minor = 5;
   // gpuSpecs.totalGlobalMem = 12079136768;
    gpuSpecs.totalGlobalMem = 5032706048;
  int myDevice;

  cudaGetDevice(&myDevice);

  printf("The ID of the current GPU: %d\n", myDevice);

  cudaChooseDevice(&myDevice, &gpuSpecs);

  cudaSetDevice(myDevice);
  printf("Total Global Memory: %ld\n", gpuSpecs.totalGlobalMem);

  // let's use the device to do some calculations

  int c;
  int *d_c;

  cudaMalloc((void **)&d_c, sizeof(int)); // Be careful not to dereference this
                                          // pointer, attach d_ to varibles

  myKernel << <1, 1>>> (2, 7, d_c); // Be careful with the syntax one less "<"
                                    // you are in trouble!

  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost); // memcpy(dest,src,...

  printf(" The value of a + b = %d\n", c);
  printf(" CUDA rocks!\n");

  cudaFree(d_c);

  return 0;
}
