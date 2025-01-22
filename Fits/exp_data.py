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

mE   = 547.862e6/(1e3)
mEp  = 957.78e6/(1e3)


#====== Gammas ======#
GammaB0s = (6.582119569e-16/(1527e-15))/(1e3)
GammaB0  = (6.582119569e-16/(1519e-15))/(1e3)
GammaBp  = (6.582119569e-16/(1638e-15))/(1e3)

#========== Factors ===========#
fBpK0bKp  = (1/(2*mBp))*np.sqrt((mBp - mK0b - mKp)*(mBp + mK0b + mKp)*(mBp + mK0b - mKp)*(mBp - mK0b + mKp))
fBpP0Pp   = (1/(2*mBp))*np.sqrt((mBp - mP0 - mPp)*(mBp + mP0 + mPp)*(mBp + mP0 - mPp)*(mBp - mP0 + mPp))
fBpEPp    = (1/(2*mBp))*np.sqrt((mBp - mE - mPp)*(mBp + mE + mPp)*(mBp + mE - mPp)*(mBp - mE + mPp))
fBpEpPp   = (1/(2*mBp))*np.sqrt((mBp - mEp - mPp)*(mBp + mEp + mPp)*(mBp + mEp - mPp)*(mBp - mEp + mPp))
fB0K0K0b  = (1/(2*mB0))*np.sqrt((mB0 - mK0 - mK0b)*(mB0 + mK0 + mK0b)*(mB0 + mK0 - mK0b)*(mB0 - mK0 + mK0b))
fB0PpPm   = (1/(2*mB0))*np.sqrt((mB0 - mPm - mPp)*(mB0 + mPm + mPp)*(mB0 + mPm - mPp)*(mB0 - mPm + mPp))
fB0KpKm   = (1/(2*mB0))*np.sqrt((mB0 - mKm - mKp)*(mB0 + mKm+ mKp)*(mB0 + mKm - mKp)*(mB0 - mKm + mKp))
fB0P0P0   = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mP0)*(mB0 + mP0 + mP0)*(mB0 + mP0 - mP0)*(mB0 - mP0 + mP0))
fB0P0E    = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mE)*(mB0 + mP0 + mE)*(mB0 + mP0 - mE)*(mB0 - mP0 + mE))
fB0EE     = (1/(2*mB0))*np.sqrt((mB0 - mE - mE)*(mB0 + mE + mE)*(mB0 + mE - mE)*(mB0 - mE + mE))
fB0sPpKm  = (1/(2*mB0s))*np.sqrt((mB0s - mPp - mKm)*(mB0s + mPp + mKm)*(mB0s + mPp - mKm)*(mB0s - mPp + mKm))
fB0sP0K0b = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mK0b)*(mB0s + mP0 + mK0b)*(mB0s + mP0 - mK0b)*(mB0s - mP0 + mK0b))
fB0sEK0b  = (1/(2*mB0s))*np.sqrt((mB0s - mE - mK0b)*(mB0s + mE + mK0b)*(mB0s + mE - mK0b)*(mB0s - mE + mK0b))

fBpPpK0   = (1/(2*mBp))*np.sqrt((mBp - mPp - mK0)*(mBp + mPp + mK0)*(mBp + mPp - mK0)*(mBp - mPp + mK0))
fBpP0Kp   = (1/(2*mBp))*np.sqrt((mBp - mP0 - mKp)*(mBp + mP0 + mKp)*(mBp + mP0 - mKp)*(mBp - mP0 + mKp))
fBpEKp    = (1/(2*mBp))*np.sqrt((mBp - mE - mKp)*(mBp + mE + mKp)*(mBp + mE - mKp)*(mBp - mE + mKp))
fB0PmKp   = (1/(2*mB0))*np.sqrt((mB0 - mPm - mKp)*(mB0 + mPm + mKp)*(mB0 + mPm - mKp)*(mB0 - mPm + mKp))
fB0P0K0   = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mK0)*(mB0 + mP0 + mK0)*(mB0 + mP0 - mK0)*(mB0 - mP0 + mK0))
fB0EK0    = (1/(2*mB0))*np.sqrt((mB0 - mE - mK0)*(mB0 + mE + mK0)*(mB0 + mE - mK0)*(mB0 - mE + mK0))
fB0sK0K0b = (1/(2*mB0s))*np.sqrt((mB0s - mK0 - mK0b)*(mB0s + mK0 + mK0b)*(mB0s + mK0 - mK0b)*(mB0s - mK0 + mK0b))
fB0sPpPm  = (1/(2*mB0s))*np.sqrt((mB0s - mPm - mPp)*(mB0s + mPm + mPp)*(mB0s + mPm - mPp)*(mB0s - mPm + mPp))
fB0sKpKm  = (1/(2*mB0s))*np.sqrt((mB0s - mKm - mKp)*(mB0s + mKm+ mKp)*(mB0s + mKm - mKp)*(mB0s - mKm + mKp))
fB0sP0P0  = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mP0)*(mB0s + mP0 + mP0)*(mB0s + mP0 - mP0)*(mB0s - mP0 + mP0))
fB0sP0E   = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mE)*(mB0s + mP0 + mE)*(mB0s + mP0 - mE)*(mB0s - mP0 + mE))
fB0sEE    = (1/(2*mB0s))*np.sqrt((mB0s - mE - mE)*(mB0s + mE + mE)*(mB0s + mE - mE)*(mB0s - mE + mE))

fBpKpEp   = (1/(2*mBp))*np.sqrt((mBp - mKp - mEp)*(mBp + mKp + mEp)*(mBp + mKp - mEp)*(mBp - mKp + mEp))
fB0P0Ep   = (1/(2*mB0))*np.sqrt((mB0 - mP0 - mEp)*(mB0 + mP0 + mEp)*(mB0 + mP0 - mEp)*(mB0 - mP0 + mEp))
fB0EEp    = (1/(2*mB0))*np.sqrt((mB0 - mE - mEp)*(mB0 + mE + mEp)*(mB0 + mE - mEp)*(mB0 - mE + mEp))
fB0sEpK0b = (1/(2*mB0s))*np.sqrt((mB0s - mEp - mK0b)*(mB0s + mEp + mK0b)*(mB0s + mEp - mK0b)*(mB0s - mEp + mK0b))

fBpEpKp   = (1/(2*mBp))*np.sqrt((mBp - mEp - mKp)*(mBp + mEp + mKp)*(mBp + mEp - mKp)*(mBp - mEp + mKp))
fB0EpK0   = (1/(2*mB0))*np.sqrt((mB0 - mEp - mK0)*(mB0 + mEp + mK0)*(mB0 + mEp - mK0)*(mB0 - mEp + mK0))
fB0sP0Ep  = (1/(2*mB0s))*np.sqrt((mB0s - mP0 - mEp)*(mB0s + mP0 + mEp)*(mB0s + mP0 - mEp)*(mB0s - mP0 + mEp))
fB0sEEp   = (1/(2*mB0s))*np.sqrt((mB0s - mE - mEp)*(mB0s + mE + mEp)*(mB0s + mE - mEp)*(mB0s - mE + mEp))

fB0EpEp   = (1/(2*mB0))*np.sqrt((mB0 - mEp - mEp)*(mB0 + mEp + mEp)*(mB0 + mEp - mEp)*(mB0 - mEp + mEp))
fB0sEpEp  = (1/(2*mB0s))*np.sqrt((mB0s - mEp - mEp)*(mB0s + mEp + mEp)*(mB0s + mEp - mEp)*(mB0s - mEp + mEp))


# Defining the experimental results (29)

#========== Decay rates (28) ===========#
BpK0bKp_exp, BpK0bKp_inc = 1.31e-6, 0.14e-6
BpP0Pp_exp, BpP0Pp_inc   = 5.31e-6, 0.26e-6
B0K0K0b_exp, B0K0K0b_inc = 1.21e-6, 0.16e-6
B0PpPm_exp, B0PpPm_inc   = 5.43e-6, 0.26e-6
B0KpKm_exp, B0KpKm_inc   = 8.2e-8, 1.5e-8
B0P0P0_exp, B0P0P0_inc   = 1.55e-6, 0.17e-6
B0sPpKm_exp, B0sPpKm_inc = 5.90e-6, 0.7e-6

BpEPp_exp, BpEPp_inc     = 4.02e-6,0.27e-6
BpEpPp_exp, BpEpPp_inc   = 2.7e-6, 0.9e-6
B0P0E_exp, B0P0E_inc     = 0.41e-6, 0.17e-6
B0P0Ep_exp, B0P0Ep_inc   = 1.2e-6, 0.6e-6
B0EE_exp, B0EE_inc       = 0.5e-6, 0.32e-6
B0EpEp_exp, B0EpEp_inc   = 0.6e-6, 0.64e-6

BpPpK0_exp, BpPpK0_inc   = 2.39e-5, 0.06e-5
BpP0Kp_exp, BpP0Kp_inc   = 1.320e-5, 0.04e-5
B0PmKp_exp, B0PmKp_inc   = 2e-5, 0.04e-5
B0P0K0_exp, B0P0K0_inc   = 1.01e-5, 0.04e-5
B0sK0K0b_exp,B0sK0K0b_inc= 1.76e-5, 0.31e-5
B0sPpPm_exp, B0sPpPm_inc = 7.2e-7, 1e-7
B0sKpKm_exp, B0sKpKm_inc = 2.72e-5, 0.23e-5
B0sP0P0_exp, B0sP0P0_inc = 2.8e-6, 2.8e-6

BpEKp_exp, BpEKp_inc     = 2.4e-6, 0.4e-6
BpEpKp_exp, BpEpKp_inc   = 70.4e-6, 2.5e-6
B0EK0_exp, B0EK0_inc     = 1.23e-6, 0.27e-6
B0EpK0_exp, B0EpK0_inc   = 66e-6, 4e-6
B0sEE_exp, B0sEE_inc     = 100e-6, 107e-6
B0sEEp_exp, B0sEEp_inc   = 25e-6, 23e-6
B0sEpEp_exp, B0sEpEp_inc = 33e-6, 7e-6

#========== Direct CP asymmetries (16) ==========#
ACP_BpK0bKp_exp, ACP_BpK0bKp_inc = 0.04, 0.14
ACP_BpP0Pp_exp, ACP_BpP0Pp_inc   = -0.01, 0.04
ACP_B0K0K0b_exp, ACP_B0K0K0b_inc = 0.06, 0.26 #double check
ACP_B0PpPm_exp, ACP_B0PpPm_inc   = 0.314, 0.03
ACP_B0P0P0_exp, ACP_B0P0P0_inc   = 0.30, 0.20
ACP_B0sPpKm_exp, ACP_B0sPpKm_inc = 0.224, 0.012

ACP_BpEPp_exp, ACP_BpEPp_inc     = -0.14, 0.07
ACP_BpEpPp_exp, ACP_BpEpPp_inc   = 0.06, 0.16

ACP_BpPpK0_exp, ACP_BpPpK0_inc   = -0.003, 0.015
ACP_BpP0Kp_exp, ACP_BpP0Kp_inc   = 0.027, 0.012
ACP_B0PmKp_exp, ACP_B0PmKp_inc   = -0.0831, 0.0031
ACP_B0P0K0_exp, ACP_B0P0K0_inc   = 0.00, 0.08
ACP_B0sKpKm_exp, ACP_B0sKpKm_inc = -0.162, 0.035

ACP_BpEKp_exp, ACP_BpEKp_inc     = -0.37, 0.08
ACP_BpEpKp_exp, ACP_BpEpKp_inc   = 0.004, 0.011
ACP_B0EpK0_exp, ACP_B0EpK0_inc   = 0.06, 0.04

#=========== Indirect CP asymmetries (5) ===========#
SCP_B0PpPm_exp, SCP_B0PpPm_inc = -0.670, 0.03
SCP_B0K0K0b_exp, SCP_B0K0K0b_inc = -1.08,0.49 #double check
SCP_B0P0K0_exp, SCP_B0P0K0_inc = -0.64, 0.13 #Change of sign because of the final state is CP odd
SCP_B0sKpKm_exp, SCP_B0sKpKm_inc = 0.14, 0.03
SCP_B0EpK0_exp, SCP_B0EpK0_inc = -0.63, 0.06 # I believe the sign is correct, but needs to be checked


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