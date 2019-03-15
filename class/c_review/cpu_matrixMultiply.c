/*
 * Purpose: Demonstrate and time matrix multiplication on the CPU
 *
 * Date and time: 04/09/2014
 * Last modified: 03/16/2016
 * Author: Inanc Senocak
 *
 * to compile: gcc -O2 -o matMult.exe matrixMultiply.c
 * to execute: ./matMult.exe
 *
 */

#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#define SIZE 50
#define A_NROW SIZE
#define A_NCOL SIZE
#define B_NROW A_NCOL
#define B_NCOL SIZE
#define C_NROW A_NROW
#define C_NCOL B_NCOL

void printMatrix(float *c)
{
    int i, j, idx;

    int nrow = 6; // C_NROW
    int ncol = 6; // C_NCOL

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            idx = j + i * ncol;
            printf("%8.2f ; ", c[ idx ]);
        }
        printf("\n");
    }
    printf("\n");
}
void InitializeMatrices(float *a, float *b)
{
    int i, j, k, idx;

    // initialize matrices a & b
    for (i = 0; i < A_NROW; i++) {
        for (k = 0; k < A_NCOL; k++) {
            idx      = k + i * A_NCOL;
            a[ idx ] = ( float ) idx;
        }
    }

    for (k = 0; k < B_NROW; k++) {
        for (j = 0; j < B_NCOL; j++) {
            idx      = j + k * B_NCOL;
            b[ idx ] = ( float ) idx;
        }
    }
}

void matrixMultiply(float *a, float *b, float *c)
{
    // this function does the following matrix multiplication c = a * b
    // a(i x k); b(k x j); c(i x j)

    int   i, j, k, idx;
    float sum = 0.f;
    // multiply the matrices C=A*B
    for (i = 0; i < A_NROW; i++) {
        for (j = 0; j < B_NCOL; j++) {
            for (k = 0; k < A_NCOL; k++) {
                sum += a[ k + i * A_NCOL ] * b[ j + k * B_NCOL ];
            }
            c[ j + i * C_NCOL ] = sum;
            sum                 = 0.f;
        }
    }
}

int main(int argc, char *argv[])
{
    float *a = malloc(sizeof(*a) * A_NROW * A_NCOL);
    float *b = malloc(sizeof(*b) * B_NROW * B_NCOL);
    // c = (float *)malloc(sizeof(float)*C_NROW*C_NCOL);
    float *c = calloc(C_NROW * C_NCOL, sizeof(*c));

    InitializeMatrices(a, b);

    printf("=====MATRIX A=====\n");
    printMatrix(a);

    printf("=====MATRIX B=====\n");
    printMatrix(b);

    double start, finish, elapsedTime;

    GET_TIME(start);

    matrixMultiply(a, b, c);

    GET_TIME(finish);

    printf("=====MATRIX C=====\n");
    printMatrix(c);

    elapsedTime = finish - start;

    printf("elapsed wall time = %.3f microseconds\n", elapsedTime * 1.0e6);
    // printf("# of FLOP = %f \n", flop);
    // printf("GFLOPS = %.6f \n", (flop / elapsedTime)*1.e-9);

    // printf("Size = %i, steps = %i, wallTime = %ld, CPU Time = %f\n", NX, nTimeSteps, wallTime,
    // CPUTime);

    free(a);
    free(b);
    free(c);

    return (0);
}
