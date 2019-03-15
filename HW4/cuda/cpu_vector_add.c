/*
 * Simple CPU program to add two long vectors
 *
 * Author: Inanc Senocak
 *
 * compile: gcc -std=c99 -O2 -lm cpu_vector_add.c -o time_CPU.exe
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"

void addArrays(int n, double *a, double *b, double *c) 
{
	for (int i = 0; i<n; i++)
	c[i] = a[i] + b[i];
}
int main(void)
{
	int n=100000000;
	double start, finish, flop, elapsedTime;
	
	double *x = malloc( n * sizeof *x);
	double *y = malloc( n * sizeof *y);
	double *z = malloc( n * sizeof *z);

	for (int i = 0; i < n; i++){
		x[i] = 3.5;
		y[i] = 1.5;
	}

	GET_TIME(start);
	addArrays(n, x, y, z);
	GET_TIME(finish);
	elapsedTime = finish-start;
	printf("z[100] = %4.2f\n",z[100]);
	printf("elapsed wall time = %0.6f seconds \n",elapsedTime);
	free(x);
	free(y);
	free(z);

return EXIT_SUCCESS;
}
