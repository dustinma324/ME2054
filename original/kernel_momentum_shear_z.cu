/** ------------------------------------------------------------------------ **/
/**                      MOMENTUM WITH EFFECTIVE VISCOSITY                   **/
/** ------------------------------------------------------------------------ **/

#include "schumann.h"
#include <stdio.h>

/*---------------------------------------------------------------------------*/
/*-----------------------  Instantaneous Z momentum   -----------------------*/
/*---------------------------------------------------------------------------*/
__global__ void momentum_schumann_z(int const sections, int const time_method, const REAL *d_u,
                                    const REAL *d_v, const REAL *d_w, REAL *d_wnew, REAL *d_wt1,
                                    REAL *d_wt2, const REAL *d_nu, const REAL *d_nu_rans,
                                    const REAL *d_tauw31w, const REAL *d_tauw31e,
                                    const REAL *d_tauw32s, const REAL *d_tauw32n, const REAL *d_phi,
                                    const REAL *d_Tinflow, const REAL *d_df,
                                    const REAL *d_forcing_z)
{
    REAL dt = DCONSTANT_DT;
    //   REAL H = DCONSTANT_TURB_TRANS;
    //   int const pid = DCONSTANT_DEVICE;
    //   int const nproc = DCONSTANT_GPUCOUNT;
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
    REAL wproj;
#endif // VELOCITY_NUDGING

    // REAL Dc  = 0.000001;
    // REAL Dt  = 0.000001;
    // REAL De  = 0.000001;
    // REAL Dte = 0.000001;
    // REAL Dw  = 0.000001;
    // REAL Dtw = 0.000001;
    // REAL Dn  = 0.000001;
    // REAL Dtn = 0.000001;
    // REAL Ds  = 0.000001;
    // REAL Dts = 0.000001;

    REAL vip = NU; // visc. plus one index
    REAL vic = NU; // visc. at c
    REAL vxe = NU; // visc. x-dir east
    REAL vxw = NU; // visc. x-dir west
    REAL vyn = NU; // visc. y-dir north
    REAL vys = NU; // visc. y-dir south

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

    // REAL nu_t_ramp = tinflow_check_xdir *

#ifdef TEMPERATURE_SOLUTION
    REAL t_at_w = DCONSTANT_TEMP_ISOTH;
#endif // TEMPERATURE_SOLUTION
#ifdef TEMP_TURB_INLET
    REAL ti_at_w = DCONSTANT_TEMP_ISOTH;
#endif // TEMP_TURB_INLET

    while (k < kend) {
        unsigned int base = I + k * NXNY;
        unsigned int top  = base + NXNY;
        REAL         z    = (REAL)(k + DCONSTANT_ZFIRST) * DCONSTANT_DZ;
        // Shear stresses
        REAL tauw32s = d_tauw32s[ k * NX + xpos ]; // south wall
        REAL tauw32n = d_tauw32n[ k * NX + xpos ]; // north wall
        REAL tauw31w = d_tauw31w[ k * NY + ypos ]; // west wall
        REAL tauw31e = d_tauw31e[ k * NY + ypos ]; // east wall

        REAL diff, adv;
        REAL boussinesq_approx = 0.0;

        REAL uc  = d_u[ base ];
        REAL uw  = d_u[ base - 1 ];
        REAL ut  = d_u[ top ];
        REAL utw = d_u[ base + NXNY - 1 ];

        REAL vc  = d_v[ base ];
        REAL vs  = d_v[ base - NX ];
        REAL vt  = d_v[ top ];
        REAL vts = d_v[ base + NXNY - NX ];

        REAL wc = d_w[ base ];
        REAL we = d_w[ base + 1 ];
        REAL ww = d_w[ base - 1 ];
        REAL wn = d_w[ base + NX ];
        REAL ws = d_w[ base - NX ];
        REAL wt = d_w[ top ];
        REAL wb = d_w[ base - NXNY ];

        // Average the turbulent viscosities around the w velocity face
        if (d_nu != 0) {
            // if ( d_nu_rans != 0 )
            //{
            //   Dc  = d_df[base]; // The allocation logic should guarantee that d_df is allocated
            //   if hybrid RANS/LES is chosen Dt  = d_df[top]; De  = d_df[base+1]; Dte =
            //   d_df[base+NX+1]; Dw  = d_df[base-1]; Dtw = d_df[base+NX-1]; Dn  = d_df[base+NXNY];
            //   Dtn = d_df[base+NXNY+NX];
            //   Ds  = d_df[base-NXNY];
            //   Dts = d_df[base-NXNY+NX];

            //   vip = NU + d_nu_rans[top] * ( Dn <= H ) + d_nu[top] * ( Dn > H );
            //   vic = NU + d_nu_rans[base] * ( Dc <= H ) + d_nu[base] * ( Dc > H );
            //   if ( ( Dc > H ) || ( Dt > H ) || ( De > H ) || ( Dte > H ) ) {
            //      vxe = NU + 0.25*(d_nu[base]+ d_nu[top] + d_nu[base+1] + d_nu[top+1]);
            //   } else {
            //      vxe = NU + 0.25*(d_nu_rans[base] + d_nu_rans[top] + d_nu_rans[base+1] +
            //      d_nu_rans[top+1]);
            //   }
            //   if ( ( Dc > H ) || ( Dt > H ) || ( Dw > H ) || ( Dtw > H ) ) {
            //      vxw = NU + 0.25*(d_nu[base] + d_nu[top] + d_nu[base-1] + d_nu[top-1]);
            //   } else {
            //      vxw = NU + 0.25*(d_nu_rans[base] + d_nu_rans[top] + d_nu_rans[base-1] +
            //      d_nu_rans[top-1]);
            //   }
            //   if ( ( Dc > H ) || ( Dt > H ) || ( Dtn > H ) || ( Dn > H ) ) {
            //      vyn = NU + 0.25*(d_nu[base] + d_nu[top] + d_nu[base+NX] + d_nu[top+NX]);
            //   } else {
            //      vyn = NU + 0.25*(d_nu_rans[base] + d_nu_rans[top] + d_nu_rans[base+NX] +
            //      d_nu_rans[top+NX]);
            //   }
            //   if ( ( Dc > H ) || ( Dt > H ) || ( Dts > H ) || ( Ds > H ) ) {
            //      vys = NU + 0.25*(d_nu[base] + d_nu[top] + d_nu[base-NX] + d_nu[top-NX]);
            //   } else {
            //      vys = NU + 0.25*(d_nu_rans[base] + d_nu_rans[top] + d_nu_rans[base-NX] +
            //      d_nu_rans[top-NX]);
            //   }
            //} else
            //{
            // Eliminate eddy viscosity in perturbation zone
            REAL nu_perturb = (1.0 - eddyvisc_check);
            vic             = nu_perturb * d_nu[ base ]; // use to reduce global memory accesses
            vip             = nu_perturb * d_nu[ top ];  // use to reduce global memory accesses
            vxe = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + 1 ] + d_nu[ top + 1 ]);
            vxw = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base - 1 ] + d_nu[ top - 1 ]);
            vyn = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NX ] + d_nu[ top + NX ]);
            vys = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base - NX ] + d_nu[ top - NX ]);
            vic += NU; // Add molecular viscosity
            vip += NU; // Add molecular viscosity
                       //}
        }

#ifdef TEMP_TURB_INLET
        // Apply boussinesq approximation for turbulent inflow temperature bouyancy effects
        if (tinflow_check) {
            // average the temperature about the w velocity point
            ti_at_w = 0.5 * (d_Tinflow[ base ] + d_Tinflow[ base + NXNY ]);
        }

        // For a channel flow with wall-normal direction in z, we enforce symmetry about the
        // centerline in the temperature perturbation inflow method by ensuring the direction of the
        // gravity vector is toward the wall
        // REAL chan_symm = DCONSTANT_PERTURB_CHANNEL_SYMM * ( ( nproc == 1 && k < nlayers / 2 ) ||
        //                                                    ( nproc > 1  && pid < nproc / 2 ) );
        // boussinesq_approx += boussinesq_bouyancy( (1.0 - 2.0*chan_symm) * DCONSTANT_GRAVITY_Z,
        // ti_at_w);
        boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_Z, ti_at_w);
// if ( xpos == 8 && ypos == 60 ) {
//   printf("xyz = %d %d %d  tlocal = %f bouy = %f vic = %f\n", xpos, ypos, k, ti_at_w,
//   boussinesq_approx, vic);
//}
#endif // TEMP_TURB_INLET

        // Calculate shear terms
        REAL tau31p = vxe * ((we - wc) * dxi + (ut - uc) * dzi);
        REAL tau31m = vxw * ((wc - ww) * dxi + (utw - uw) * dzi);
        REAL tau32p = vyn * ((wn - wc) * dyi + (vt - vc) * dzi);
        REAL tau32m = vys * ((wc - ws) * dyi + (vts - vs) * dzi);
        REAL tau33p = 2.0 * vip * (wt - wc) * dzi;
        REAL tau33m = 2.0 * vic * (wc - wb) * dzi;

        // Apply instantaneous shear BC

        // u-component averaged to w-component
        REAL uaw = 0.25 * (uc + uw + ut + utw);
        // v-component averaged to w-component
        REAL vaw = 0.25 * (vc + vs + vt + vts);
        // east/west wall-parallel velocity magnitude
        REAL magyz = sqrt(vaw * vaw + wc * wc);
        // north/south wall-parallel velocity magnitude
        REAL magxz = sqrt(uaw * uaw + wc * wc);

        // Avoid division by zero if velocities are zero
        magxz = (magxz < MACHEPS) + (magxz > MACHEPS) * magxz;
        magyz = (magyz < MACHEPS) + (magyz > MACHEPS) * magyz;

        REAL temp;
        // west wall
        temp
        = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_W) == BOUNDARY_INSTASHEAR) && (xpos == 1));
        tau31m = (1.0 - temp) * tau31m + temp * (tauw31w * (wc / magyz));

        // east wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_E) == BOUNDARY_INSTASHEAR)
                      && (xpos == (DCONSTANT_NX - 2)));
        tau31p = (1.0 - temp) * tau31p - temp * (tauw31e * (wc / magyz));

        // south wall
        temp
        = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_S) == BOUNDARY_INSTASHEAR) && (ypos == 1));
        tau32m = (1.0 - temp) * tau32m + temp * (tauw32s * (wc / magxz));

        // north wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_N) == BOUNDARY_INSTASHEAR)
                      && (ypos == (DCONSTANT_NY - 2)));
        tau32p = (1.0 - temp) * tau32p - temp * (tauw32n * (wc / magxz));

        // Calculate viscous diffusion term
        diff = dxi * (tau31p - tau31m) + dyi * (tau32p - tau32m) + dzi * (tau33p - tau33m);

        // Calculate convection term
        adv = 0.25
              * (dxi * ((uc + ut) * (wc + we) - (uw + utw) * (ww + wc))
                 + dyi * ((vc + vt) * (wc + wn) - (vs + vts) * (ws + wc))
                 + dzi * ((wc + wt) * (wc + wt) - (wb + wc) * (wb + wc)));

        // Apply upwind scheme if desired
        adv += UPWIND * 0.25
               * (dxi * ((fabs(uc + ut) * (wc - we)) - (fabs(uw + utw) * (ww - wc)))
                  + dyi * ((fabs(vc + vt) * (wc - wn)) - (fabs(vs + vts) * (ws - wc)))
                  + dzi * ((fabs(wc + wt) * (wc - wt)) - (fabs(wb + wc) * (wb - wc))));

#ifdef TEMPERATURE_SOLUTION
        // Apply boussinesq approximation for physical temperature bouyancy effects
        if (d_phi != 0) {
            // average the temperature about the w velocity point
            t_at_w = 0.5 * (d_phi[ base ] + d_phi[ base + NXNY ]);

            boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_Z, t_at_w);
        }
#endif // TEMPERATURE_SOLUTION

        REAL start          = 0.50;
        REAL end            = 0.90;
        REAL thickness      = end - start;
        REAL damping_factor = 0.5 * (z > start * DCONSTANT_LZ) * (z < end * DCONSTANT_LZ)
                              * pow((z - start * DCONSTANT_LZ) / (thickness * DCONSTANT_LZ), 5)
                              + 0.5 * (z > end * DCONSTANT_LZ);
        REAL rayleigh_damping = -damping_factor * wc;

#ifdef VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_wt2[ base ], d_wt1[ base ], d_wnew[ base ], wc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

#else // VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_wt2[ base ], d_wt1[ base ], wproj, wc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

        // Apply forcing
        if (d_forcing_z != 0) wproj += dt * d_forcing_z[ base ];

        d_wnew[ base ] = wproj;

#endif // VELOCITY_NUDGING

        if ((!do_mid) && (k == kbeg)) {
            k = nlayers - 1;
        } else {
            k++;
        }
    }
}
