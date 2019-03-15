/* Calculation of the infinite series of exp(x)
 *
 * Author: Dustin
 * ME2045
 * October 5th, 2018
 *
 * Use the following to compile:
 * gcc -o infinit_series.exe -std=c99 -O3 -lm infinit_series.c
 * clang -o infinit_series.exe -lm infinit_series.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

int main(){
// Specifying User Input Parameter
float x;	//single precision
double xd;	//double precision
printf("Enter 2 of the same number:\n");
scanf("%f%lf",&x,&xd);

// Global Variable
int i; 

// FLOAT - Single Precision
float e, fact, increment;
float tol = 1e-8f;	//f after value specifu storing as single precision
fact = 1.0f;		// "	
e = 1.0f;		// "

for( i = 1; i < ULONG_MAX ; i++ ){
	fact = fact * i;			//factorial calculation
	increment = pow(fabsf(x),i) / fact;	//broken up for stopping criterion
	e = e + increment;			//infinite series
	
	if (increment < tol){
	        if ( x < 0 ){	//to accurately calculate the negative x value
       		e = 1/e;
        	}
	printf("Single Precision: x = %1.0f, e = %1.6f, exp(x) = %1.6f\n",x,e,exp(x));
	break;
	}
}

// DOUBLE - Double Precision
double c, factd, incrementd;
double told = 1e-8;	// not putting d after number stores as double
factd = 1;
c = 1;

for( i = 1; i < ULONG_MAX ; i++ ){
	factd = factd * i;		//factorial calculation
        incrementd = pow(fabs(xd),i) / factd;	//broken up for stopping criterion
	c = c + incrementd;		//infinite series

	if (incrementd < told){
		if ( xd < 0 ){   //to accurately calculate the negative x value
                c = 1/c;
                }
	printf("Double Precision: x = %1.0lf, e = %1.15lf, exp(x) = %1.15lf\n",xd,c,exp(xd));
	break;
	}
}
return EXIT_SUCCESS;
}
