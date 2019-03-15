#include "meshGrid.h"

void meshGrid ( REAL *x )
{
    INT i;
    for (i=0; i<NX; i++) {
        x[i] =  DX * ( (REAL) i );
    }
}

