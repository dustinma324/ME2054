#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <array>

#define RADIUS 3
#define NX 20

void stencil_cpu(float *in, float *out){
	int i,j;
	float sum = 0.0f;
	for (int i = RADIUS; i < sizeof(in)-RADIUS; i++){
        	for(int j = -RADUIS; j<= RADIUS; j++){
		sum += in[i+j];
		}
	out[i] = sum;
	}
}

int main(){
float *A = cudaMallocManaged(NX*(sizeof *A));
float *tmp = cudeMallocManaged(NX*(sizeof *tmp)); // will only have sizeof(A) - 2(radius) elements

stencil_cpu(A,tmp);

return EXIT_SUCCESS;
}
 
