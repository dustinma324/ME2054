/*
 * Purpose: Demonstrate matrix multiplication using OpenACC
 *
 * Date and time: 04/05/2016
 * Author: Inanc Senocak
 *
 * to compile: pgcc -fast -acc -ta=nvidia,time -Minfo=accel -o matrixMult.exe acc_matrixMultiply.c
 * to time the code: export PGI_ACC_TIME=1
 * to execute the code: ./executableName
 *
 */

#include <math.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000
#define A_NROW SIZE
#define A_NCOL SIZE
#define B_NROW A_NCOL
#define B_NCOL SIZE
#define C_NROW A_NROW
#define C_NCOL B_NCOL
#define RANDOM_MAX 4

void printMatrix(float *c);
void initializeMatrices(restrict float *a, restrict float *b);
void matrixMultiply(const restrict float *a, const restrict float *b, restrict float *c);

int main(int argc, char *argv[])
{
    float *a, *b, *c;

    a = ( float * ) calloc(A_NROW * A_NCOL, sizeof(float));
    b = ( float * ) calloc(B_NROW * B_NCOL, sizeof(float));
    c = ( float * ) calloc(C_NROW * C_NCOL, sizeof(float));

    int myDevice = 0;
    acc_init(acc_device_nvidia);
    acc_set_device_num(myDevice, acc_device_nvidia);
    int gpuCount = acc_get_num_devices(acc_device_nvidia);
    printf("#GPUs on the system: %d\n", gpuCount);

    initializeMatrices(a, b);
    printf("=====MATRIX A=====\n");
    printMatrix(a);
    printf("=====MATRIX B=====\n");
    printMatrix(b);
#pragma acc data pcopyin(a [0:A_NROW * A_NCOL], b [0:B_NROW * B_NCOL]),pcopyout(c [0:C_NROW * C_NCOL])
    {
        matrixMultiply(a, b, c);
    }
    printf("=====MATRIX C=====\n");
    printMatrix(c);

    free(a);
    free(b);
    free(c);

    return (0);
} // end main

float random_number(int n, int seed)
{
    int i, j;
    srand(( unsigned ) seed);
    float r = (rand( ) % (n + 1));
    return r;
}

void printMatrix(float *c)
{
    int i, j, idx;
    int n_display = 10;
    int nrow      = n_display;
    int ncol      = n_display;
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            idx = j + i * ncol;
            printf("%8.2f ; ", c[ idx ]);
        }
        printf("\n");
    }
    printf("\n");
}
void initializeMatrices(float *a, float *b)
{
    int i, j, k, idx;
    for (i = 0; i < A_NROW; i++) {
        for (k = 0; k < A_NCOL; k++) {
            idx      = k + i * A_NCOL;
            a[ idx ] = random_number(RANDOM_MAX, idx);
        }
    }

    for (k = 0; k < B_NROW; k++) {
        for (j = 0; j < B_NCOL; j++) {
            idx      = j + k * B_NCOL;
            b[ idx ] = random_number(RANDOM_MAX,idx);
        }
    }
}

void matrixMultiply(const restrict float *a, const restrict float *b, restrict float *c)
{
    // this function does the following matrix multiplication c = a * b
    // a(i x k); b(k x j); c(i x j)

    int i, j, k, idx;

    // multiply the matrices C=A*B
    float sum = 0.f;
#pragma acc data present(a, b) // is this line really needed?
    {
#pragma acc parallel loop gang vector_length(256)
        for (i = 0; i < A_NROW; i++) {
#pragma acc loop vector
            for (j = 0; j < B_NCOL; j++) {
                sum = 0.f;
#pragma acc loop reduction(+ : sum)
                for (k = 0; k < A_NCOL; k++) {
                    sum += a[ k + i * A_NCOL ] * b[ j + k * B_NCOL ];
                }
                c[ j + i * C_NCOL ] = sum;
            }
        }
    }
}
