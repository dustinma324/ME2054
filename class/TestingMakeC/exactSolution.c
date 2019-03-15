#include "exactSolution.h"

void exactSolution( REAL *uExact, const REAL *x )
{
    INT i;
    for (i=0; i<NX; i++) {
        uExact[i] = x[i]*x[i]*exp(-x[i]);
    }
}

