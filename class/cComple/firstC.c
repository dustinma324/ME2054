/* This will be my first test on finding the convergence rage of 1/k
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
float i;
float k;
float tol;

k = 0.0f;
tol = 1e-6f;

for (i = 1; i < ULONG_MAX; i++){
	k = k + 1/i;
	printf("i = %10.1f, k = %4.8f\n",i,k);

/*	
	if(abs(1/i)<tol){
	printf("Convergence Reached! i = %10.1f, k = %4.8f\n",i,k);
	break;
	}
*/
}
	return EXIT_SUCCESS;
}
