import numpy as np
import matplotlib.pyplot as plt
import math
from exp_data import *

root2 = np.sqrt(2)
root3 = np.sqrt(3)
root6 = np.sqrt(6)

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

    #===Diagrams with strong phases===#
    TT = abs(ampT)
    CC = abs(ampC)*np.exp(1j*delC)
    PPuc= abs(ampP)*np.exp(1j*delP)
    AA = abs(ampA)*np.exp(1j*delA)
    PAuc = abs(ampPA)*np.exp(1j*delPA)
    PPtc = abs(ampP2)*np.exp(1j*delP2)
    PAtc = abs(ampPA2)*np.exp(1j*delPA2)
    
    #===Diagrams with weak phases===#
    #===Delta S = 0===#
    TTd = TT*Vubd
    CCd = CC*Vubd
    PPucd = PPuc*Vubd
    AAd = AA*Vubd
    PAucd = PAuc*Vubd
    PPtcd = PPtc*Vtbd
    PAtcd = PAtc*Vtbd

    #===EWP Diagrams===#
    PEWTd = -3/4((c9+c10)/(c1+c2)*(TT+CC+AA)+(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubd
    PEWCd = -3/4((c9+c10)/(c1+c2)*(TT+CC-AA)-(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubd
    PEWAd = -3/2(c9+c10)/(c1+c2)*AA*Vubd

    #===Delta S = 1===#
    TTs = TT*Vubs
    CCs = CC*Vubs
    PPucs = PPuc*Vubs
    AAs = AA*Vubs
    PAucs = PAuc*Vubs
    PPtcs = PPtc*Vtbs
    PAtcs = PAtc*Vtbs

    #===EWP Diagrams===#
    PEWTs = -3/4((c9+c10)/(c1+c2)*(TT+CC+AA)+(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubs
    PEWCs = -3/4((c9+c10)/(c1+c2)*(TT+CC-AA)-(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubs
    PEWAs = -3/2(c9+c10)/(c1+c2)*AA*Vubs

    #===Diagrams with conjugate weak phases===#
    TTdb = TT*Vubdb
    CCdb = CC*Vubdb
    PPucdb = PPuc*Vubdb
    AAdb = AA*Vubdb
    PAucdb = PAuc*Vubdb
    PPtcdb = PPtc*Vtbdb
    PAtcdb = PAtc*Vtbdb

    #===EWP Diagrams===#
    PEWTdb = -3/4((c9+c10)/(c1+c2)*(TT+CC+AA)+(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubdb
    PEWCdb = -3/4((c9+c10)/(c1+c2)*(TT+CC-AA)-(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubdb
    PEWAdb = -3/2(c9+c10)/(c1+c2)*AA*Vubdb

    #===Delta S = 1===#
    TTsb = TT*Vubsb
    CCsb = CC*Vubsb
    PPucsb = PPuc*Vubsb
    AAsb = AA*Vubsb
    PAucsb = PAuc*Vubsb
    PPtcsb = PPtc*Vtbsb
    PAtcsb = PAtc*Vtbsb

    #===EWP Diagrams===#
    PEWTsb = -3/4((c9+c10)/(c1+c2)*(TT+CC+AA)+(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubsb
    PEWCsb = -3/4((c9+c10)/(c1+c2)*(TT+CC-AA)-(c9-c10)/(c1-c2)*(TT-CC-AA))*Vubsb
    PEWAsb = -3/2(c9+c10)/(c1+c2)*AA*Vubsb

    #===Diagram contributions to amplitudes===#

    DIAGd0 =   [[0,0,1,1,0,1,0,0,-1/3,0],\
                [-1/root2,-1/root2,0,0,0,0,0,-1/root2,-1/root2,0],\
                [-1/root6,-1/root6,-2/root6,-2/root6,0,-2/root6,0,-1/root6,-1/(3*root6),0,-4/(3*root6)],\
                [0,0,1,0,1,1,1,0,-1/3,-2/3],\
                [-1,0,-1,0,-1,-1,-1,0,-2/3,-1/3],\
                [0,0,0,0,-1,0,-1,0,0,-1/3],\
                [0,-1/root2,1/root2,0,1/root2,1/root2,1/root2,-1/root2,-1/(3*root2),1/(3*root2)],\
                [0,0,-1/root3,0,0,-1/root3,0,0,1/(3*root3),1/root3],\
                [0,1/(3*root2),1/(3*root2),0,1/root2,1/(3*root2),1/root2,1/(3*root2),-1/(9*root2),-1/(3*root2)],\
                [-1,0,-1,0,0,-1,0,0,-2/3,0],\
                [0,-1/root2,1/root2,0,0,1/root2,0,-1/root2,-1/(3*root2),0],\
                [0,-1/root6,1/root6,0,0,1/root6,0,-1/root6,-1/(3*root6),0]]
    DIAGd1 =   [[0,0,1,1,0,1,0,0,-1/3,0],\
                [-1/root2,-1/root2,-1/root2,-1/root2,0,-1/root2,0,-1/root2,-root2/3,0],\
                [-1/root6,-1/root6,1/root6,1/root6,0,1/root6,0,-1/root6,-4/(3*root6),0],\
                [-1,0,-1,0,0,-1,0,0,-2/3,0],\
                [0,-1/root2,-1/root2,-1/root2,-1/root2,0,-1/root2,0,-1/root2,-1/(3*root2),0],\
                [0,-1/root6,1/root6,0,0,1/root6,0,-1/root6,-1/(3*root6),0],\
                [0,0,1,0,1,1,1,0,-1/3,-2/3],\
                [0,0,0,0,-1,0,-1,0,0,-1/3],\
                [-1,0,-1,0,-1,-1,-1,0,-2/3,-1/3],\
                [0,0,0,0,1/root2,0,1/root2,0,0,1/(3*root2)],\
                [0,-1/root3,0,0,0,0,0,-1/root3,0,1/root3],\

