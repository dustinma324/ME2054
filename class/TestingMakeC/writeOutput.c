#include "writeOutput.h"

void writeOutput ( const REAL *x, const REAL *uExact, const REAL *u)
{
    INT i;
    FILE *output;
    output=fopen("1d_heat.dat","w");

    for (i=0; i<NX; i++) {
        fprintf( output, "%10f %10f %10f\n", x[i], uExact[i], u[i] );
    }
    fclose(output);
}

