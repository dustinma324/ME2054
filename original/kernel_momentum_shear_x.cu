/** ------------------------------------------------------------------------ **/
/**                      MOMENTUM WITH EFFECTIVE VISCOSITY                   **/
/** ------------------------------------------------------------------------ **/

#include "schumann.h"

/*---------------------------------------------------------------------------*/
/*-----------------------  Instantaneous X momentum  ------------------------*/
/*---------------------------------------------------------------------------*/
__global__ void momentum_schumann_x(int const sections, int const time_method, const REAL *d_u,
                                    const REAL *d_v, const REAL *d_w, REAL *d_unew, REAL *d_ut1,
                                    REAL *d_ut2, const REAL *d_nu, const REAL *d_nu_rans,
                                    const REAL *d_tauw12s, const REAL *d_tauw12n,
                                    const REAL *d_tauw13b, const REAL *d_tauw13t, const REAL *d_phi,
                                    const REAL *d_Tinflow, const REAL *d_df,
                                    const REAL *d_forcing_x, REAL *d_tauw13bst, REAL *d_tauw13tst
#ifdef SHEAR_OUTPUT_TAU13
                                    ,
                                    REAL *d_mod13
#endif // SHEAR_OUTPUT_TAU13
)
{
    REAL dt = DCONSTANT_DT;
    //   REAL H = DCONSTANT_TURB_TRANS;
    int const nlayers = DCONSTANT_NLAYERS;

    unsigned int xpos = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ypos = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int I    = ypos * (gridDim.x * blockDim.x) + xpos + NXNY;

    if ((xpos == 0) || (xpos >= (DCONSTANT_NX - 1)) || (ypos == 0) || (ypos >= (DCONSTANT_NY - 1))
        || (sections == 0))
        return;

    unsigned int kbeg, kend;
    if (sections & SECTION_BOT) {
        kbeg = 0;
    } else {
        kbeg = 1;
    }
    if (sections & SECTION_TOP) {
        kend = nlayers;
    } else {
        kend = nlayers - 1;
    }
    bool do_mid = (sections & SECTION_MID);
    int  k      = kbeg;

#ifndef VELOCITY_NUDGING
    REAL uproj;
#endif // VELOCITY_NUDGING

    // REAL Dc  = 0.000001;
    // REAL De  = 0.000001;
    // REAL Dn  = 0.000001;
    // REAL Dne = 0.000001;
    // REAL Ds  = 0.000001;
    // REAL Dse = 0.000001;
    // REAL Dt  = 0.000001;
    // REAL Dte = 0.000001;
    // REAL Db  = 0.000001;
    // REAL Dbe = 0.000001;

    REAL vip = NU; // visc. plus one index
    REAL vic = NU; // visc. at c
    REAL vyn = NU; // visc. y-dir north
    REAL vys = NU; // visc. y-dir south
    REAL vzt = NU; // visc. z-dir top
    REAL vzb = NU; // visc. z-dir bottom
                   // if((I-NXNY)<100)printf("Small index =%i\n",I-NXNY);
    // else if((I-NXNY)>5000)printf("Large index =%i\n",I-NXNY);

    // Set storage to zero
    d_tauw13bst[ I - NXNY ] = 0.0;
    d_tauw13tst[ I - NXNY ] = 0.0;
// if((I-NXNY)<0)printf("negative index =%\n"i, I-NXNY);
#ifdef TEMP_TURB_INLET_PARTIAL_COUPLING
    int tinflow_check_xdir = (xpos > DCONSTANT_PERTURB_GCMIN_X && xpos < DCONSTANT_PERTURB_GCMAX_X
                              && ypos > DCONSTANT_PERTURB_GCMIN_Y);
    int tinflow_check_ydir = (ypos > DCONSTANT_PERTURB_GCMIN_Y && ypos < DCONSTANT_PERTURB_GCMAX_Y
                              && xpos > DCONSTANT_PERTURB_GCMIN_X);
    int tinflow_check      = (d_Tinflow != 0) && ((tinflow_check_xdir || tinflow_check_ydir));
#else
    int tinflow_check = (d_Tinflow != 0);
#endif // TEMP_TURB_INLET_PARTIAL_COUPLING
    int eddyvisc_check
    = (xpos < DCONSTANT_PERTURB_EDDYVIS_MAX_X || ypos < DCONSTANT_PERTURB_EDDYVIS_MAX_Y);

#ifdef TEMPERATURE_SOLUTION
    REAL t_at_u = DCONSTANT_TEMP_ISOTH;
#endif // TEMPERATURE_SOLUTION
#ifdef TEMP_TURB_INLET
    REAL ti_at_u = DCONSTANT_TEMP_ISOTH;
#endif // TEMP_TURB_INLET

    while (k < kend) {
        unsigned int base = I + k * NXNY;
        unsigned int east = base + 1;
        REAL         z    = (REAL)(k + DCONSTANT_ZFIRST) * DCONSTANT_DZ;

        // Shear stresses
        REAL tauw13b = d_tauw13b[ I - NXNY ];      // bottom wall
        REAL tauw13t = d_tauw13t[ I - NXNY ];      // top wall
        REAL tauw12s = d_tauw12s[ k * NX + xpos ]; // south wall
        REAL tauw12n = d_tauw12n[ k * NX + xpos ]; // north wall

        REAL diff, adv;
        REAL boussinesq_approx = 0.0;

        REAL uc = d_u[ base ];
        REAL ue = d_u[ east ];
        REAL uw = d_u[ base - 1 ];
        REAL un = d_u[ base + NX ];
        REAL us = d_u[ base - NX ];
        REAL ut = d_u[ base + NXNY ];
        REAL ub = d_u[ base - NXNY ];

        REAL vc  = d_v[ base ];
        REAL ve  = d_v[ east ];
        REAL vs  = d_v[ base - NX ];
        REAL vse = d_v[ base - NX + 1 ];

        REAL wc  = d_w[ base ];
        REAL we  = d_w[ east ];
        REAL wb  = d_w[ base - NXNY ];
        REAL wbe = d_w[ base - NXNY + 1 ];

        // Average the turbulent viscosities around the u velocity face
        if (d_nu != 0) {
            // if ( d_nu_rans != 0 )
            //{
            //   Dc  = d_df[base]; // The allocation logic should guarantee that d_df is allocated
            //   if hybrid RANS/LES is chosen De  = d_df[east]; Dn  = d_df[base+NX]; Dne =
            //   d_df[base+NX+1]; Ds  = d_df[base-NX]; Dse = d_df[base-NX+1]; Dt  = d_df[base+NXNY];
            //   Dte = d_df[base+NXNY+1];
            //   Db  = d_df[base-NXNY];
            //   Dbe = d_df[base-NXNY+1];

            //   // If any cell centers lie above the hybrid RANS/LES transition hieght,
            //   // use the SGS eddy viscosity values
            //   vip = NU + d_nu_rans[east] * ( De <= H ) + d_nu[east] * ( De > H );
            //   vic = NU + d_nu_rans[base] * ( Dc <= H ) + d_nu[base] * ( Dc > H );
            //   if ( ( Dc > H ) || ( De > H ) || ( Dne > H ) || ( Dn > H ) ) {
            //      vyn = NU + 0.25*(d_nu[base]+ d_nu[east] + d_nu[base+NX+1] + d_nu[base+NX]);
            //   } else {
            //      vyn = NU + 0.25*(d_nu_rans[base] + d_nu_rans[east] + d_nu_rans[base+NX+1] +
            //      d_nu_rans[base+NX]);
            //   }
            //   if ( ( Dc > H ) || ( De > H ) || ( Dse > H ) || ( Ds > H ) ) {
            //      vys = NU + 0.25*(d_nu[base] + d_nu[east] + d_nu[base-NX+1] + d_nu[base-NX]);
            //   } else {
            //      vys = NU + 0.25*(d_nu_rans[base] + d_nu_rans[east] + d_nu_rans[base-NX+1] +
            //      d_nu_rans[base-NX]  );
            //   }
            //   if ( ( Dc > H ) || ( De > H ) || ( Dte > H ) || ( Dt > H ) ) {
            //      vzt = NU + 0.25*(d_nu[base] + d_nu[east] + d_nu[base+NXNY+1] + d_nu[base+NXNY]);
            //   } else {
            //      vzt = NU + 0.25*(d_nu_rans[base] + d_nu_rans[east] + d_nu_rans[base+NXNY+1] +
            //      d_nu_rans[base+NXNY]);
            //   }
            //   if ( ( Dc > H ) || ( De > H ) || ( Dbe > H ) || ( Db > H ) ) {
            //      vzb = NU + 0.25*(d_nu[base] + d_nu[east] + d_nu[base-NXNY+1] + d_nu[base-NXNY]);
            //   } else {
            //      vzb = NU + 0.25*(d_nu_rans[base] + d_nu_rans[east] + d_nu_rans[base-NXNY+1] +
            //      d_nu_rans[base-NXNY]);
            //   }
            //} else
            //{
            // Eliminate eddy viscosity in perturbation zone
            REAL nu_perturb = (1.0 - eddyvisc_check);
            vip             = nu_perturb * d_nu[ east ]; // use to reduce global memory accesses
            vic             = nu_perturb * d_nu[ base ]; // use to reduce global memory accesses
            vyn = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NX + 1 ] + d_nu[ base + NX ]);
            vys = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base - NX + 1 ] + d_nu[ base - NX ]);
            vzt
            = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NXNY + 1 ] + d_nu[ base + NXNY ]);
            vzb
            = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base - NXNY + 1 ] + d_nu[ base - NXNY ]);
            vip += NU; // Add molecular viscosity
            vic += NU; // Add molecular viscosity
                       //}
        }

#ifdef TEMP_TURB_INLET
        // Apply boussinesq approximation for turbulent inflow temperature bouyancy effects
        if (tinflow_check) {
            // average the temperature about the w velocity point
            ti_at_u = 0.5 * (d_Tinflow[ base ] + d_Tinflow[ base + 1 ]);
        }

        boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_X, ti_at_u);
#endif // TEMP_TURB_INLET

        // Calculate modeled and viscous shear stress terms
        REAL tau11p = 2.0 * vip * (ue - uc) * dxi;
        REAL tau11m = 2.0 * vic * (uc - uw) * dxi;
        REAL tau12p = vyn * ((un - uc) * dyi + (ve - vc) * dxi);
        REAL tau12m = vys * ((uc - us) * dyi + (vse - vs) * dxi);
        REAL tau13p = vzt * ((ut - uc) * dzi + (we - wc) * dxi);
        REAL tau13m = vzb * ((uc - ub) * dzi + (wbe - wb) * dxi);

        // Apply instantaneous shear BC

        // v-component averaged to u-component
        REAL vau = 0.25 * (vc + ve + vs + vse);
        // w-component averaged to u-component
        REAL wau = 0.25 * (wc + we + wb + wbe);
        // top/bottom wall-parallel velocity magnitude
        REAL magxy = sqrt(uc * uc + vau * vau);
        // north/south wall-parallel velocity magnitude
        REAL magxz = sqrt(uc * uc + wau * wau);

        // Avoid division by zero if velocities are zero
        magxy = (magxy < MACHEPS) + (magxy > MACHEPS) * magxy;
        magxz = (magxz < MACHEPS) + (magxz > MACHEPS) * magxz;

        REAL temp;

        // south wall
        temp
        = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_S) == BOUNDARY_INSTASHEAR) && (ypos == 1));
        tau12m = (1.0 - temp) * tau12m + temp * (tauw12s * (uc / magxz));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw12s = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw12s, tauw12s * ( uc / magxz ));

        // north wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_N) == BOUNDARY_INSTASHEAR)
                      && (ypos == (DCONSTANT_NY - 2)));
        tau12p = (1.0 - temp) * tau12p - temp * (tauw12n * (uc / magxz));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw12n = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw12n, tauw12n * ( uc / magxz ));

        // bottom wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_B) == BOUNDARY_INSTASHEAR)
                      && (DCONSTANT_DEVICE == 0) && (k == 1));
        tau13m = (1.0 - temp) * tau13m + temp * (tauw13b * (uc / magxy));
        // if ( temp > 0.0 && ypos == 32 ) printf("[ %d %d %d ] tauw13b = %.15f schu =
        // %.15f\n",xpos, ypos, k, tauw13b, tauw13b * ( uc / magxy ));

        // Only add tau13m when we are on the correct indices
        d_tauw13bst[ I - NXNY ] += tau13m * temp;

        // top wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_T) == BOUNDARY_INSTASHEAR)
                      && (DCONSTANT_DEVICE == (DCONSTANT_GPUCOUNT - 1)) && (k == (nlayers - 2)));
        tau13p = (1.0 - temp) * tau13p - temp * (tauw13t * (uc / magxy));
        // if ( temp > 0.0 && ypos == 32 ) printf("[ %d %d %d ] tauw13t = %.15f schu =
        // %.15f\n",xpos, ypos, k, tauw13t, tauw13t * ( uc / magxy ));

        // Only add tau13p when we are on the correct indices
        d_tauw13tst[ I - NXNY ] += tau13p * temp;

#ifdef SHEAR_OUTPUT_TAU13
        if (d_mod13 != 0) {
            d_mod13[ base ] = tau13p;
            if (k == 1) d_mod13[ base - NXNY ] = tau13m;
        }
#endif // SHEAR_OUTPUT_TAU13

        // Calculate diffusion
        diff = dxi * (tau11p - tau11m) + dyi * (tau12p - tau12m) + dzi * (tau13p - tau13m);

        // Calculate convection term
        adv = 0.25
              * (dxi * ((uc + ue) * (uc + ue) - (uw + uc) * (uw + uc))
                 + dyi * ((vc + ve) * (uc + un) - (vs + vse) * (us + uc))
                 + dzi * ((wc + we) * (uc + ut) - (wb + wbe) * (ub + uc)));

        // apply upwind scheme if desired
        adv += UPWIND * 0.25
               * (dxi * ((fabs(uc + ue) * (uc - ue)) - (fabs(uw + uc) * (uw - uc)))
                  + dyi * ((fabs(vc + ve) * (uc - un)) - (fabs(vs + vse) * (us - uc)))
                  + dzi * ((fabs(wc + we) * (uc - ut)) - (fabs(wb + wbe) * (ub - uc))));

#ifdef TEMPERATURE_SOLUTION
        // Apply boussinesq approximation for physical temperature bouyancy effects
        if (d_phi != 0) {
            // average the temperature about the w velocity point
            t_at_u = 0.5 * (d_phi[ base ] + d_phi[ base + 1 ]);

            boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_X, t_at_u);
        }

#endif // TEMPERATURE_SOLUTION
        // REAL coriolis = DCONSTANT_EKMAN_ANGULARVELOCITY*vc;

        REAL start          = 0.5;
        REAL end            = 0.9;
        REAL thickness      = end - start;
        REAL damping_factor = 0.5 * (z > start * DCONSTANT_LZ) * (z < end * DCONSTANT_LZ)
                              * pow((z - start * DCONSTANT_LZ) / (thickness * DCONSTANT_LZ), 5)
                              + 0.5 * (z > end * DCONSTANT_LZ);
        REAL rayleigh_damping = -damping_factor * (uc - 10.84);

#ifdef VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_ut2[ base ], d_ut1[ base ], d_unew[ base ], uc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

#else // VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_ut2[ base ], d_ut1[ base ], uproj, uc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

        // Apply forcing
        if (d_forcing_x != 0) uproj += dt * d_forcing_x[ base ];

        d_unew[ base ] = uproj;

#endif // VELOCITY_NUDGING

        if ((!do_mid) && (k == kbeg)) {
            k = nlayers - 1;
        } else {
            k++;
        }
    }
}
