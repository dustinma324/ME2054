 /* Finding the machine epsilon in single and double precision
 *
 * Author: Dustin
 * ME2045
 * September 21
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

int main(){

        float epsilon = 1.0f;
	double epsilond = 1.0;	

// Single Precision
        while (epsilon+1.0f != 1.0f){
	epsilon = epsilon/2.0f;
        }

//double Precision
        while (epsilond+1.0 != 1.0){
        epsilond = epsilond/2.0;
        }

// Output Final Machine Epsilon
	printf("Single Precision: Machine Epsilon = %0.10e\n",epsilon);
        printf("Double Precision: Machine Epsilon = %0.10e\n",epsilond);
        return EXIT_SUCCESS;
}

