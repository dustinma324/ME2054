// OPENACC: pgcc -acc -Minfo=accel -ta=nvidia -o saxpy_acc saxpy.c 
// GCC: gcc -O3 -lm -o saxpy_acc saxpy.c 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <openacc.h>
/*
void saxpy(int n, float a, float *x, float *restrict y)
{
#pragma acc kernels
 for (int i = 0; i < n; ++i)
 y[i] = a*x[i] + y[i];
}
*/
int main(int argc, char **argv)
{
	int N = 1000;
	if (argc >1)
		N = atoi(argv[1]);

	float *A = (float *) calloc(N*N, sizeof(*A));
	float *Anew = (float *) calloc(N*N, sizeof(*Anew));


	for (int i = 0; i < N; ++i) {
		for(int j = 0; j < N; ++j){
		A[i+j*N] = 2.0f;
		Anew[i+j*N] = 1.0f;
		}
	} 

	float tol = 1e-6;
	int iter = 0;
	int iter_max = 100;
	int n = N;
	int m = N;
	float err;

	while ( err > tol && iter < iter_max ) {
	err=0.0;

	#pragma acc kernels 	
	for( int j = 1; j < n-1; j++) {
		for(int i = 1; i < m-1; i++) {
		Anew[j*N+i] = 0.25 * (A[j*N+(i+1)] + A[j*N+(i-1)] +
		A[(j-1)*N+i] + A[(j+1)*N+i]);
		err = fmax(err, abs(Anew[j*N+i] - A[j*N+i]));
		}
	}

	#pragma acc kernels
	for( int j = 1; j < n-1; j++) {
		for( int i = 1; i < m-1; i++ ) {
		A[j*N+i] = Anew[j*N+i];
		}
	}

 iter++;
}  

/*
	// Perform SAXPY on 1M elements
	saxpy(N, 3.0f, A, Anew);
*/
return EXIT_SUCCESS;
}
