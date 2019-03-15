/* Finding the largest possible integer by CRC' H2P cluster
 *
 *
 * Author: Dustin
 * ME2045
 * September 21
 *
 * Use the following to compile
 * gcc largestInt.c -std=c99 -O3 -lm -o largestInt.exe
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

int main(){
	unsigned int i,j;
	i = 2147300000;
	j = 0;
	while (i-j != 0){
		j = i;
/*
		if (i < 0){
		printf("i = %d \n",i);
		return 0;
		}

		if(remainder(i,100000) == 0){
		printf("i = %d \n",i);
		}
*/		
		printf("i = %d \n",i);
		i++;
	}
	printf("The largest positive Integer = %d\n",j);
	return EXIT_SUCCESS;
}
