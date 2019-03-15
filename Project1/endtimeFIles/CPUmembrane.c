/*
 * ME 2054 Parallel Scientific Computing
 * Project 1 - Finite Difference Solution of a Vibrating 2D Membrane on a GPU
 * Due: November 6,2018
 *
 * Author: Dustin (Ting-Hsuan) Ma
 *
 * gcc -std=c99 -O2 -lm CPUmembrane.c -o CPUrun.exe
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define LX 4.0f
#define LY 2.0f
#define NX 31.0f
#define NY 21.0f
#define DX LX/(NX-1)
#define DY LY/(NY-1)
#define H DX

#define C sqrt(5.f)
#define DT 0.4f*(H/C)

#define ENDTIME 20.0f
#define M_PI 3.14159265358979323846f
#define INFINITE 200

//Making calculation part easier
#define IC i+j*NX
#define IP1 (i+1)+j*NX
#define IM1 (i-1)+j*NX
#define JP1 i+(j+1)*NX
#define JM1 i+(j-1)*NX

typedef float REAL;
typedef int INT;

void calculatingWave(REAL *now, REAL *old, REAL *out)
{
        INT i,j,ic,ip1,im1,jp1,jm1;
        for (j = 1; j < NY - 1; j++) {
                for (i = 1; i < NX - 1; i++) {
                ic = IC;
                ip1 = IP1;
                im1 = IM1;
                jp1 = JP1;
                jm1 = JM1;
                out[ic] = 2.f * now[ic] - old[ic] + ((C*C*DT*DT)/(H*H)) * (now[ip1] + now[im1] + now[jp1] + now[jm1] - 4.f * now[ic]);
                }
        }
}

void initializeMatrices(REAL *in)
{
	INT i, j, idx;
	REAL x, y = 0.0f;

	for (j = 0; j < NY; j++) {
	x = 0.0f;
		for (i = 0; i < NX; i++) {
		idx = IC;
		// Eq 2.
		in[idx] = 0.1f * ( 4.f * x - (x * x) ) * ( 2.f * y - (y * y) );
		x += DX;
		}
	y += DY;
	}
}

void applyingBoundary(REAL *in)
{
        INT i, j, idx;
        for (j = 0; j < NY; j++) {
                for (i = 0; i < NX; i++) {
                idx = IC;
		//Eq 4 - 7
                if (i == 0 || i == NX) in[idx] = 0.0f;
		if (j == 0 || j == NY) in[idx] = 0.0f;
		}
        }

}

void initializeSolution(REAL *in, REAL *out)
{
	INT i,j,ic,ip1,im1,jp1,jm1;
        for (j = 1; j < NY - 1; j++) {
                for (i = 1; i < NX - 1; i++) {
                ic = IC;
		ip1 = IP1;
		im1 = IM1;
		jp1 = JP1;
		jm1 = JM1;
		out[ic] = in[ic] + (0.5 * (C*C*DT*DT)/(H*H)) * (in[ip1] + in[im1] + in[jp1] + in[jm1] - 4.f*in[ic]);
                }
        }
}

void analyticalSolution(REAL *out){

	INT idx, i, j;
	REAL x, m, n, y = 0;

	for (j = 0; j < NY; j++) {
	x = 0.f;
		for (i = 0; i < NX; i++) {
                idx = IC;
			for (m = 1.f; m <= INFINITE; m += 2.f){
				for (n = 1.f; n <= INFINITE; n += 2.f){
				out[idx] += 0.426050f/(m*m*m*n*n*n)*cos((ENDTIME+DT)*sqrt(5.f)*M_PI/4.f*sqrt(m*m+4.f*n*n))*sin(m*M_PI*x/4.f)*sin(n*M_PI*y/2.f);
				}
			}
		x += DX;
		}
	y += DY;
	}

}

void	outputMatrix(REAL *in){

// I dont like the format it outputs, for matching purposes, I turned it off.
//FILE *output;
//output = fopen("2d_membrane_initial.dat","w");

        INT i,j,idx;
        for (j = 0; j < NY; j++) {
                for (i = 0; i < NX; i++) {
                idx = i + j * NX;
                printf("%6.3f ",in[idx]);
		//fprintf(output," %8.4f",in[idx]);
                }
	printf("\n");
        }
printf("\n");
//fclose(output);
}

INT main(){

	// CFL check
	if (sqrt(C) * DT / H < 1.0f){
	printf("CFL condition is met\n");
	}
	
	else{
	printf("CFL condition is not met, try again\n");
	return EXIT_SUCCESS;
	}	
	
	// Allocating memory for CPU
	REAL *phi, *phi_old, *phi_new, *tmp;
	REAL *Exact_phi;
	
	Exact_phi = calloc (NX*NY, sizeof(*Exact_phi));
	
	phi = calloc(NX * NY, sizeof(*phi));
	phi_old = calloc(NX * NY, sizeof(*phi_old));
	phi_new = calloc(NX * NY, sizeof(*phi_new));

	// Calculating Analitical Solution
	analyticalSolution(Exact_phi);
	printf("===============================Exact Solution===============================\n");
	outputMatrix(Exact_phi);

	// Initializing Mesh, boundaries, and solution
	initializeMatrices(phi);
	applyingBoundary(phi);
	initializeSolution(phi,phi_old);

	// Solving time function until endtime is reached
	REAL time = 0.0f;
	while (time < ENDTIME){
		calculatingWave(phi,phi_old,phi_new);	

		tmp = phi;
		phi = phi_new;
		phi_new = phi_old;
		phi_old = tmp;
	
		time += DT;
	}
	
	// Outputing the contents of matrix	
	printf("===============================Final Solution CPU===============================\n");
	outputMatrix(phi);
	
	// Deallocating memory
	free(phi);free(phi_old);free(phi_new);free(Exact_phi);
	phi=NULL; phi_old=NULL;phi_new=NULL;Exact_phi=NULL;

return EXIT_SUCCESS;
}

