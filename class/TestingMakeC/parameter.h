#ifndef PARAMETER_H
#define PARAMETER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include "timer.h"

#define LX 3.0f    /* length of the domain in x-direction  */
#define NX 32    /* includes boundary points on both end */
#define DX        LX / ( (REAL) (NX-1) )
#define ALPHA     1.0f
#define DT        0.25f * DX*DX / ALPHA

#ifndef SINGLE
typedef double REAL;
typedef int   INT;
#define PI M_PI
#else
typedef float REAL;
typedef int    INT;
#define PI M_PI
#endif

#endif
