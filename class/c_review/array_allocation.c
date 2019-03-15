/* this code illustrate two approaches to allocate dynamic memory
 * for two dimensional arrays a[ ][ ]
 *
 * to compile: gcc -O2 -std=c99 -o allocate.exe array_allocation.c
 * to execute: ./allocate.exe
 */

#include <stdio.h>
#include <stdlib.h>

int main( )
{
    int nr = 3, nc = 4, i, j, idx, count;

    /* A single pointer to store a 2D array. Flattened, 1D storage */

    int *arr1 = ( int * ) malloc(nr * nc * sizeof(int));

    for (i = 0; i < nr; i++) {
        for (j = 0; j < nc; j++) {
            idx           = i * nc + j;
            *(arr1 + idx) = ++count;
        }
    }

    printf("printing arr1:\n");
    for (i = 0; i < nr; i++){
        for (j = 0; j < nc; j++){
            idx = i * nc + j;
            printf("%d ", arr1[ idx ]);
        }
    }

    printf("\n----------------\n");

    /* Using an array of pointers (so-called pointer to a pointer) to store a 2D array*/

    int **arr2 = ( int ** ) malloc(nr * sizeof(int *));
    for (i = 0; i < nr; i++)
        arr2[ i ] = ( int * ) malloc(nc * sizeof(int));

    // Note that arr2[i][j] is same as *(*(arr2+i)+j)
    count = 0;
    for (i = 0; i < nr; i++)
        for (j = 0; j < nc; j++)
            arr2[ i ][ j ] = ++count; // OR *(*(arr2+i)+j) = ++count

    printf("printing arr2:\n");
    for (i = 0; i < nr; i++)
        for (j = 0; j < nc; j++)
            printf("%d ", arr2[ i ][ j ]);

    printf("\n-----------------\n");

    return EXIT_SUCCESS;
}
