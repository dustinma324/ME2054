/*
 * Simple CPU program to add two long vectors
 *
 * Author: Inanc Senocak
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>

void addArrays(int n, double *a, double *b, double *c) 
{
  for (int i = 0; i<n; i++)
      c[i] = a[i] + b[i];
}

int main(void)
{

    int n = 10000000;

    double *x = malloc( n * sizeof *x);
    double *y = malloc( n * sizeof *y);
    double *z = malloc( n * sizeof *z);

    for (int i = 0; i < n; i++){
        x[i] = 3.5;
        y[i] = 1.5;
    }

    addArrays(n, x, y, z);

    printf("z[100] = %4.2f\n",z[100]);

    free(x);
    free(y);
    free(z);

    return EXIT_SUCCESS;
}
