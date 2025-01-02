import numpy as np
import matplotlib.pyplot as plt
import math
from exp_data import *

def chi2(ampT,ampC,ampP,ampA,ampPA,ampP2,ampPA2,delC,delP,delA,delPA,delP2,delPA2,Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas):
    #===CKM elements===#
    Vubd = Vub*Vud*np.exp(1j*gamma)
    Vtbd = Vtb*Vtd*np.exp(-1j*beta)
    Vubs = Vub*Vus*np.exp(1j*gamma)
    Vtbs = -Vtb*Vts*np.exp(-1j*betas)

    #===Conjugate CKM elements===#
    Vubdb = Vub*Vud*np.exp(-1j*gamma)
    Vtbdb = Vtb*Vtd*np.exp(1j*beta)
    Vubsb = Vub*Vus*np.exp(-1j*gamma)
    Vtbsb = -Vtb*Vts*np.exp(1j*betas)

    #===Diagrams with strong+CKM phases===#
    TT = abs(ampT)
    CC = abs(ampC)*np.exp(1j*delC)
    PPuc= abs(ampP)*np.exp(1j*delP)
    AA = abs(ampA)*np.exp(1j*delA)
    PAuc = abs(ampPA)*np.exp(1j*delPA)
    PPtc = abs(ampP2)*np.exp(1j*delP2)
    PAtc = abs(ampPA2)*np.exp(1j*delPA2)
    
    