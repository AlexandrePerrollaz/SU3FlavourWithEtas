import numpy as np
import math

alpha = 1/129
c1,c2,c9,c10 = 1.144,-0.308,-1.280*alpha,0.328*alpha

mB0s = 5.36692e9/(1e3)
mB0  = 5.27965e9/(1e3)
mBp  = 5.27934e9/(1e3)

mPp  = 139.57e6/(1e3)
mPm  = 139.57e6/(1e3)
mP0  = 134.9768e6/(1e3)

mKm  = 493.677e6/(1e3)
mKp  = 493.677e6/(1e3)
mK0  = 497.611e6/(1e3)
mK0b = 497.611e6/(1e3)

#====== Gammas ======#
GammaB0s = (6.582119569e-16/(1527e-15))/(1e3)
GammaB0  = (6.582119569e-16/(1519e-15))/(1e3)
GammaBp  = (6.582119569e-16/(1638e-15))/(1e3)

#========== Factors ===========#
fBpK0bKp  = (1/(2*mBp))*np.sqrt((mBp - mK0b - mKp)*(mBp + mK0b + mKp)*(mBp + mK0b - mKp)*(mBp - mK0b + mKp))
fBpP0Pp   = (1/(2*mBp))*np.sqrt((mBp - mP0 - mPp)*(mBp + mP0 + mPp)*(mBp + mP0 - mPp)*(mBp - mP0 + mPp))
fB0K0K0b  = (1/(2*mB0))*np.sqrt((mB0 - mK0 - mK0b)*(mB0 + mK0 + mK0b)*(mB0 + mK0 - mK0b)*(mB0 - mK0 + mK0b))
fB0PmPp   = (1/(2*mB0))*np.sqrt((mB0 - mPm - mPp)*(mB0 + mPm + mPp)*(mB0 + mPm - mPp)*(mB0 - mPm + mPp))
fB0KmKp   = (1/(2*mB0))*np.sqrt((mB0 - mKm - mKp)*(mB0 + mKm+ mKp)*(mB0 + mKm - mKp)*(mB0 - mKm + mKp))
fB0P0P0   = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mP0)*(mB0 + mP0 + mP0)*(mB0 + mP0 - mP0)*(mB0 - mP0 + mP0))
fB0sPpKm  = (1/(2*mB0s))*np.sqrt((mB0s - mPp - mKm)*(mB0s + mPp + mKm)*(mB0s + mPp - mKm)*(mB0s - mPp + mKm))
fB0sP0K0b = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mK0b)*(mB0s + mP0 + mK0b)*(mB0s + mP0 - mK0b)*(mB0s - mP0 + mK0b))

fBpPpK0   = (1/(2*mBp))*np.sqrt((mBp - mPp - mK0)*(mBp + mPp + mK0)*(mBp + mPp - mK0)*(mBp - mPp + mK0))
fBpP0Kp   = (1/(2*mBp))*np.sqrt((mBp - mP0 - mKp)*(mBp + mP0 + mKp)*(mBp + mP0 - mKp)*(mBp - mP0 + mKp))
fB0PmKp   = (1/(2*mB0))*np.sqrt((mB0 - mPm - mKp)*(mB0 + mPm + mKp)*(mB0 + mPm - mKp)*(mB0 - mPm + mKp))
fB0P0K0   = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mK0)*(mB0 + mP0 + mK0)*(mB0 + mP0 - mK0)*(mB0 - mP0 + mK0))
fB0sK0K0b = (1/(2*mB0s))*np.sqrt((mB0s - mK0 - mK0b)*(mB0s + mK0 + mK0b)*(mB0s + mK0 - mK0b)*(mB0s - mK0 + mK0b))
fB0sPmPp  = (1/(2*mB0s))*np.sqrt((mB0s - mPm - mPp)*(mB0s + mPm + mPp)*(mB0s + mPm - mPp)*(mB0s - mPm + mPp))
fB0sKmKp  = (1/(2*mB0s))*np.sqrt((mB0s - mKm - mKp)*(mB0s + mKm+ mKp)*(mB0s + mKm - mKp)*(mB0s - mKm + mKp))
fB0sP0P0  = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mP0)*(mB0s + mP0 + mP0)*(mB0s + mP0 - mP0)*(mB0s - mP0 + mP0))


# Defining the experimental results (29)

#========== Decay rates (15) ===========#
BpK0bKp_exp, BpK0bKp_inc = 1.31e-6, 0.14e-6
BpP0Pp_exp, BpP0Pp_inc   = 5.59e-6, 0.31e-6
B0K0K0b_exp, B0K0K0b_inc = 1.21e-6, 0.16e-6
B0PmPp_exp, B0PmPp_inc   = 5.15e-6, 0.19e-6
B0KmKp_exp, B0KmKp_inc   = 8.0e-8, 1.5e-8
B0P0P0_exp, B0P0P0_inc   = 1.55e-6, 0.16e-6
B0sPpKm_exp, B0sPpKm_inc = 5.90e-6, 0.87e-6

BpPpK0_exp, BpPpK0_inc   = 2.352e-5, 0.072e-5
BpP0Kp_exp, BpP0Kp_inc   = 1.320e-5, 0.046e-5
B0PmKp_exp, B0PmKp_inc   = 1.946e-5, 0.046e-5
B0P0K0_exp, B0P0K0_inc   = 1.006e-5, 0.043e-5
B0sK0K0b_exp,B0sK0K0b_inc= 1.74e-5, 0.31e-5
B0sPmPp_exp, B0sPmPp_inc = 7.2e-7, 1.1e-7
B0sKmKp_exp, B0sKmKp_inc = 2.66e-5, 0.32e-5
B0sP0P0_exp, B0sP0P0_inc = 2.8e-6, 2.8e-6

#========== Direct CP asymmetries (11) ==========#
CBpK0bKp, CBpK0bKp_inc = 0.04, 0.14
CBpP0Pp, CBpP0Pp_inc   = 8e-3, 35e-3
CB0K0K0b, CB0K0K0b_inc = 0.06, 0.26 #double check
CB0PmPp, CB0PmPp_inc   = 0.311, 0.03
CB0P0P0, CB0P0P0_inc   = 0.30, 0.20
CB0sPpKm, CB0sPpKm_inc = 0.225, 0.012

CBpPpK0, CBpPpK0_inc   = -0.016, 0.015
CBpP0Kp, CBpP0Kp_inc   = 0.029, 0.012
CB0PmKp, CB0PmKp_inc   = -0.0836, 0.0032
CB0P0K0, CB0P0K0_inc   = -0.01, 0.10
CB0sKmKp, CB0sKmKp_inc = -0.17, 0.03

#=========== Indirect CP asymmetries (3) ===========#
SB0PmPp, SB0PmPp_inc = -0.666, 0.029
SB0K0K0b, SB0K0K0b_inc = -1.08,0.49 #double check
SB0P0K0, SB0P0K0_inc = -0.57, 0.17
SB0sKmKp, SB0sKmKp_inc = 0.14, 0.03


#=========== CKM matrix elements ===========#
Vud_exp, Vud_inc = 0.97373, 0.00031
Vus_exp, Vus_inc = 0.2243, 0.0008
Vub_exp, Vub_inc = 0.00382, 0.00020

Vtd_exp, Vtd_inc = 0.0086, 0.0002
Vts_exp, Vts_inc = 0.0415, 0.0009
Vtb_exp, Vtb_inc = 1.014,  0.029

#========== CKM phases ===========#
gamma_exp, gamma_inc = 1.1502, 0.061
beta_exp, beta_inc = 0.384, 0.013
betas_exp, betas_inc = -0.025, 0.0095