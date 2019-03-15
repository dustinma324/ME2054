/* Findind the where n will stop changing with single precision
 *
 * Author: Dustin
 * ME2045
 * September 21
 *
 * Use the following to compile
 * gcc divergentSingle.c -std=c99 -O3 -lm -o divergentSingle.exe
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"	//MAKE SURE FILE IS IN SAME DIRECTORY

int main(){
	// Timer Variables
	double start, finish, flop, elapsedTime;

	// Single Precision
	float i,k,q,increment;
	q = 0.0f;
	k = 1.0f;
	i = 1.0f;
        
	GET_TIME(start); //start the timer

	while (k-q != 0.0f){
                q = k;
                increment = 1.0f/i;
                k = k + increment;
                printf("i = %f, k = %0.25f\n",i,k);
		// when series stops increasing then it's the final iteration
		if (k-q == 0.0f){
			printf("At n = %f, the seires stops at k = %0.25f\n",i-1,q);
		break;
		}
		i++;
        }
	GET_TIME(finish);
	elapsedTime = finish - start;
	printf("elapsed wall time = %.6f seconds\n",elapsedTime);
	return EXIT_SUCCESS;
}
