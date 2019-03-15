/** ------------------------------------------------------------------------ **/
/**                      MOMENTUM WITH EFFECTIVE VISCOSITY                   **/
/** ------------------------------------------------------------------------ **/

#include "schumann.h"

/*---------------------------------------------------------------------------*/
/*-----------------------  Instantaneous Y momentum   -----------------------*/
/*---------------------------------------------------------------------------*/
__global__ void momentum_schumann_y(int const sections, int const time_method, const REAL *d_u,
                                    const REAL *d_v, const REAL *d_w, REAL *d_vnew, REAL *d_vt1,
                                    REAL *d_vt2, const REAL *d_nu, const REAL *d_nu_rans,
                                    const REAL *d_tauw21w, const REAL *d_tauw21e,
                                    const REAL *d_tauw23b, const REAL *d_tauw23t, const REAL *d_phi,
                                    const REAL *d_Tinflow, const REAL *d_df,
                                    const REAL *d_forcing_y, REAL *d_tauw23bst, REAL *d_tauw23tst)
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
    REAL vproj;
#endif // VELOCITY_NUDGING

    // REAL Dc  = 0.000001;
    // REAL Dn  = 0.000001;
    // REAL De  = 0.000001;
    // REAL Dne = 0.000001;
    // REAL Dw  = 0.000001;
    // REAL Dnw = 0.000001;
    // REAL Dt  = 0.000001;
    // REAL Dtn = 0.000001;
    // REAL Db  = 0.000001;
    // REAL Dbn = 0.000001;

    REAL vip = NU; // visc. plus one index
    REAL vic = NU; // visc. at c
    REAL vxe = NU; // visc. x-dir east
    REAL vxw = NU; // visc. x-dir west
    REAL vzt = NU; // visc. z-dir top
    REAL vzb = NU; // visc. z-dir bottom

    // Zero out storage
    d_tauw23bst[ I - NXNY ] = 0.0;
    d_tauw23tst[ I - NXNY ] = 0.0;

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
    REAL t_at_v = DCONSTANT_TEMP_ISOTH;
#endif // TEMPERATURE_SOLUTION
#ifdef TEMP_TURB_INLET
    REAL ti_at_v = DCONSTANT_TEMP_ISOTH;
#endif // TEMP_TURB_INLET

    while (k < kend) {
        unsigned int base  = I + k * NXNY;
        unsigned int north = base + NX;
        REAL         z     = (REAL)(k + DCONSTANT_ZFIRST) * DCONSTANT_DZ;
        // Shear stresses
        REAL tauw23b = d_tauw23b[ I - NXNY ];      // bottom wall
        REAL tauw23t = d_tauw23t[ I - NXNY ];      // top wall
        REAL tauw21w = d_tauw21w[ k * NY + ypos ]; // west wall
        REAL tauw21e = d_tauw21e[ k * NY + ypos ]; // east wall

        REAL diff, adv;
        REAL boussinesq_approx = 0.0;

        REAL uc  = d_u[ base ];
        REAL uw  = d_u[ base - 1 ];
        REAL un  = d_u[ north ];
        REAL unw = d_u[ base + NX - 1 ];

        REAL vc = d_v[ base ];
        REAL ve = d_v[ base + 1 ];
        REAL vw = d_v[ base - 1 ];
        REAL vn = d_v[ north ];
        REAL vs = d_v[ base - NX ];
        REAL vt = d_v[ base + NXNY ];
        REAL vb = d_v[ base - NXNY ];

        REAL wc  = d_w[ base ];
        REAL wn  = d_w[ base + NX ];
        REAL wb  = d_w[ base - NXNY ];
        REAL wbn = d_w[ base - NXNY + NX ];

        // Average the turbulent viscosities around the v velocity face
        if (d_nu != 0) {
            // if ( d_nu_rans != 0 )
            //{
            //   Dc  = d_df[base]; // The allocation logic should guarantee that d_df is allocated
            //   if hybrid RANS/LES is chosen Dn  = d_df[north]; De  = d_df[base+1]; Dne =
            //   d_df[base+NX+1]; Dw  = d_df[base-1]; Dnw = d_df[base+NX-1]; Dt  = d_df[base+NXNY];
            //   Dtn = d_df[base+NXNY+NX];
            //   Db  = d_df[base-NXNY];
            //   Dbn = d_df[base-NXNY+NX];

            //   vip = NU + d_nu_rans[north] * ( Dn <= H ) + d_nu[north] * ( Dn > H );
            //   vic = NU + d_nu_rans[base] * ( Dc <= H ) + d_nu[base] * ( Dc > H );
            //   if ( ( Dc > H ) || ( Dn > H ) || ( Dne > H ) || ( De > H ) ) {
            //      vxe = NU + 0.25*(d_nu[base]+ d_nu[north] + d_nu[base+NX+1] + d_nu[base+1]);
            //   } else {
            //      vxe = NU + 0.25*(d_nu_rans[base] + d_nu_rans[north] + d_nu_rans[base+NX+1] +
            //      d_nu_rans[base+1]);
            //   }
            //   if ( ( Dc > H ) || ( Dn > H ) || ( Dnw > H ) || ( Dw > H ) ) {
            //      vxw = NU + 0.25*(d_nu[base] + d_nu[north] + d_nu[base+NX-1] + d_nu[base-1]);
            //   } else {
            //      vxw = NU + 0.25*(d_nu_rans[base] + d_nu_rans[north] + d_nu_rans[base+NX-1] +
            //      d_nu_rans[base-1]);
            //   }
            //   if ( ( Dc > H ) || ( Dn > H ) || ( Dtn > H ) || ( Dt > H ) ) {
            //      vzt = NU + 0.25*(d_nu[base] + d_nu[north] + d_nu[base+NXNY+NX] +
            //      d_nu[base+NXNY]);
            //   } else {
            //      vzt = NU + 0.25*(d_nu_rans[base] + d_nu_rans[north] + d_nu_rans[base+NXNY+NX] +
            //      d_nu_rans[base+NXNY]);
            //   }
            //   if ( ( Dc > H ) || ( Dn > H ) || ( Dbn > H ) || ( Db > H ) ) {
            //      vzb = NU + 0.25*(d_nu[base] + d_nu[north] + d_nu[base-NXNY+NX] +
            //      d_nu[base-NXNY]);
            //   } else {
            //      vzb = NU + 0.25*(d_nu_rans[base] + d_nu_rans[north] + d_nu_rans[base-NXNY+NX] +
            //      d_nu_rans[base-NXNY]);
            //   }
            //} else
            //{
            // Eliminate eddy viscosity in perturbation zone
            REAL nu_perturb = (1.0 - eddyvisc_check);
            vic             = nu_perturb * d_nu[ base ];  // use to reduce global memory accesses
            vip             = nu_perturb * d_nu[ north ]; // use to reduce global memory accesses
            vxe = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NX + 1 ] + d_nu[ base + 1 ]);
            vxw = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NX - 1 ] + d_nu[ base - 1 ]);
            vzt
            = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base + NXNY + NX ] + d_nu[ base + NXNY ]);
            vzb
            = NU + nu_perturb * 0.25 * (vic + vip + d_nu[ base - NXNY + NX ] + d_nu[ base - NXNY ]);
            vic += NU; // Add molecular viscosity
            vip += NU; // Add molecular viscosity
                       //}
        }

#ifdef TEMP_TURB_INLET
        // Apply boussinesq approximation for turbulent inflow temperature bouyancy effects
        if (tinflow_check) {
            // average the temperature about the w velocity point
            ti_at_v = 0.5 * (d_Tinflow[ base ] + d_Tinflow[ base + NX ]);
        }

        boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_Y, ti_at_v);
#endif // TEMP_TURB_INLET

        // Calculate shear terms
        REAL tau21p = vxe * ((ve - vc) * dxi + (un - uc) * dyi);
        REAL tau21m = vxw * ((vc - vw) * dxi + (unw - uw) * dyi);
        REAL tau22p = 2.0 * vip * (vn - vc) * dyi;
        REAL tau22m = 2.0 * vic * (vc - vs) * dyi;
        REAL tau23p = vzt * ((vt - vc) * dzi + (wn - wc) * dyi);
        REAL tau23m = vzb * ((vc - vb) * dzi + (wbn - wb) * dyi);

        // Apply instantaneous shear BC

        // u-component averaged to v-component
        REAL uav = 0.25 * (uc + uw + un + unw);
        // w-component averaged to v-component
        REAL wav = 0.25 * (wc + wb + wn + wbn);
        // top/bottom wall-parallel velocity magnitude
        REAL magxy = sqrt(uav * uav + vc * vc);
        // east/west wall-parallel velocity magnitude
        REAL magyz = sqrt(wav * wav + vc * vc);

        // Avoid division by zero if velocities are zero
        magxy = (magxy < MACHEPS) + (magxy > MACHEPS) * magxy;
        magyz = (magyz < MACHEPS) + (magyz > MACHEPS) * magyz;

        REAL temp;
        // west wall
        temp
        = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_W) == BOUNDARY_INSTASHEAR) && (xpos == 1));
        tau21m = (1.0 - temp) * tau21m + temp * (tauw21w * (vc / magyz));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw21w = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw21w, tauw21w * ( vc / magyz ));

        // east wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_E) == BOUNDARY_INSTASHEAR)
                      && (xpos == (DCONSTANT_NX - 2)));
        tau21p = (1.0 - temp) * tau21p - temp * (tauw21e * (vc / magyz));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw21e = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw21e, tauw21e * ( vc / magyz ));

        // bottom wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_B) == BOUNDARY_INSTASHEAR)
                      && (DCONSTANT_DEVICE == 0) && (k == 1));
        tau23m = (1.0 - temp) * tau23m + temp * (tauw23b * (vc / magxy));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw23b = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw23b, tauw23b * ( vc / magxy ));

        // Only add tau23m when we are on the correct indices
        d_tauw23bst[ I - NXNY ] += tau23m * temp;

        // top wall
        temp   = (REAL)((GET_FACE_VALUE(DCONSTANT_FACES, FACE_T) == BOUNDARY_INSTASHEAR)
                      && (DCONSTANT_DEVICE == (DCONSTANT_GPUCOUNT - 1)) && (k == (nlayers - 2)));
        tau23p = (1.0 - temp) * tau23p - temp * (tauw23t * (vc / magxy));
        // if ( temp > 0.0 ) printf("[ %d %d %d ] tauw23t = %.15f schu = %.15f\n",xpos, ypos, k,
        // tauw23t,tauw23t * ( vc / magxy ) );

        // Only add tau23p when we are on the correct indices
        d_tauw23tst[ I - NXNY ] += tau23p * temp;

        // Calculate viscous diffusion term
        diff = dxi * (tau21p - tau21m) + dyi * (tau22p - tau22m) + dzi * (tau23p - tau23m);

        // Calculate convection term
        adv = 0.25
              * (dxi * ((uc + un) * (vc + ve) - (uw + unw) * (vw + vc))
                 + dyi * ((vc + vn) * (vc + vn) - (vs + vc) * (vs + vc))
                 + dzi * ((wc + wn) * (vc + vt) - (wb + wbn) * (vb + vc)));

        // apply upwind scheme if desired
        if (UPWIND > 0.0001) {
            adv += UPWIND * 0.25
                   * (dxi * ((fabs(uc + un) * (vc - ve)) - (fabs(uw + unw) * (vw - vc)))
                      + dyi * ((fabs(vc + vn) * (vc - vn)) - (fabs(vs + vc) * (vs - vc)))
                      + dzi * ((fabs(wc + wn) * (vc - vt)) - (fabs(wb + wbn) * (vb - vc))));
        }

#ifdef TEMPERATURE_SOLUTION
        // Apply boussinesq approximation for physical temperature bouyancy effects
        if (d_phi != 0) {
            // average the temperature about the w velocity point
            t_at_v = 0.5 * (d_phi[ base ] + d_phi[ base + NX ]);

            boussinesq_approx += boussinesq_bouyancy(DCONSTANT_GRAVITY_Y, t_at_v);
        }
#endif // TEMPERATURE_SOLUTION
        // REAL coriolis =-DCONSTANT_EKMAN_ANGULARVELOCITY*(uc-DCONSTANT_GEOSTROPHIC_WIND);
        // printf("coriolis=%f,geostrophic=%f\n",DCONSTANT_EKMAN_ANGULARVELOCITY,
        // DCONSTANT_GEOSTROPHIC_WIND);
        REAL start          = 0.50;
        REAL end            = 0.90;
        REAL thickness      = end - start;
        REAL damping_factor = 0.5 * (z > start * DCONSTANT_LZ) * (z < end * DCONSTANT_LZ)
                              * pow((z - start * DCONSTANT_LZ) / (thickness * DCONSTANT_LZ), 5)
                              + 0.5 * (z > end * DCONSTANT_LZ);
        REAL rayleigh_damping = -damping_factor * vc;

#ifdef VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_vt2[ base ], d_vt1[ base ], d_vnew[ base ], vc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

#else  // VELOCITY_NUDGING

        // Advance the time step
        PERFORM_TIME_STEP(time_method, d_vt2[ base ], d_vt1[ base ], vproj, vc,
                          (diff - adv + boussinesq_approx + rayleigh_damping));

        // Apply forcing
        if (d_forcing_y != 0) vproj += dt * d_forcing_y[ base ];

        d_vnew[ base ] = vproj;
#endif // VELOCITY_NUDGING

        if ((!do_mid) && (k == kbeg)) {
            k = nlayers - 1;
        } else {
            k++;
        }
    }
}
