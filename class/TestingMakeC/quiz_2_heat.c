/***********************************************************************
 * QUIZ INSTRUCTIONS
 *
 * you are given a partially completed code. Read the code line by line 
 * and implement the necessary steps to compute the 1d heat conduction 
 * problem given in the handout.
 *
 ***********************************************************************
 *
 * Numerical and analytical solution of the 1D heat conduction problem 
 * 
 * Author: Dustin (Ting-Hsuan) Ma
 * Date: October 19th, 2018 
 *
 * gcc -O2 -lm -std=c99 quiz_2_heat.c -o heat.exe
 *
 * to execute: ./heat.exe <simulation end time (seconds)>
 *
 *
 */

#include "meshGrid.h"
#include "exactSolution.h"
#include "solveHeat_1D.h"
#include "writeOutput.h"

INT main(INT argc, char *argv[])
{
    if (argc < 2) {
       perror("Command-line usage: executableName <end Time (seconds)>");
       exit(1);
    } 

    REAL endTime = atof(argv[1]);

	REAL *uExact, *x; 
	REAL *unew, *u, *tmp; 
//  allocate heap memory here for arrays needed in the solution algorithm

	x = calloc(NX, (sizeof *x));
	uExact = calloc(NX, (sizeof *uExact));
	unew = calloc(NX, (sizeof *unew));
        u = calloc(NX, (sizeof *u));


// calculate the x coordinates of each computational point
    meshGrid( x );
// compute the exact solution to the 1D heat conduction problem
    exactSolution ( uExact, x );

// apply boundary conditions (make sure to apply boundary conditions to both u and unew
    u[0] = 0.f;
    unew[0] = 0.f;
    unew[NX-1]=uExact[NX-1];   
    u[NX-1]=uExact[NX-1];   

    REAL time = 0.f;
    while (time < endTime) 
    {
        // call the solveHeat_1D( ) function here with correct parameters 
        // and necessary updates on the solution array
        solveHeat_1D( unew, u, x );
	tmp = u;
	u = unew;
	unew = tmp;

        time += DT;
    }

    // call the writeOutput( ) function here with correct parameters
	writeOutput ( x, uExact, u );
	// free allocated memory here
	free(x);free(uExact);free(unew);free(u);
	x=NULL;uExact=NULL;unew=NULL;u=NULL;tmp=NULL;
    return EXIT_SUCCESS;
}
