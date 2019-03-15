#include "solveHeat_1D.h"

void solveHeat_1D (REAL *unew, const REAL *u, const REAL *x)
{
    INT i;
    REAL dxi = 1.f/(DX*DX);
    REAL xc, source;

    for (i=1 ; i<NX-1 ; i++) {
        xc = x[i];
        source = -(xc*xc-4.f*xc+2.f)*exp(-xc);  //source term
        unew[i] = ( ALPHA*( u[i+1] - 2.0f*u[i] + u[i-1])*dxi + source ) * DT + u[i] ;
    }
}

