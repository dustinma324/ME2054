/* Finding the largest value that can be displayed by my home computer 
 * Author: Dustin
 * ME2045
 * September 21
 *
 * Use the following to compile
 * gcc largestHome.c -std=c99 -O3 -lm -o largestHome.exe
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

int main(){
	// Double Precision
	double i,k,q,increment;
	q = 0.0;
	k = 1.0;
	i = 1.0;
	
	// while next value subtracted by previous value doesnt equal to zero, continue
	while (k-q != 0.0){
		q = k;
		increment = 1.0/i;
		k = k + increment;

		// Output every 50,000 steps
                if(remainder(i,50000) == 0){
			printf("i = %lf, k = %0.55lf\n",i,k);
		 }

		// When divergent series stops increasing, this is the final iteration
		if (k-q == 0.0){
                        printf("At n = %lf, the seires stops increasing at k = %0.55lf\n",i-1,q);
		return 0;
                }
		i++;
	}
	return EXIT_SUCCESS;
}
