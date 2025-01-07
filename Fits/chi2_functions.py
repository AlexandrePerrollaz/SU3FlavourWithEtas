import numpy as np
import matplotlib.pyplot as plt
import math
from exp_data import *

root2 = np.sqrt(2)
root3 = np.sqrt(3)
root6 = np.sqrt(6)
Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

def amplitude_eta(amplitude_eta8, amplitude_eta1):
    theta_eta = np.arcsin(1/3)
    amplitude = amplitude_eta8*np.cos(theta_eta) - amplitude_eta1*np.sin(theta_eta)
    return amplitude

def amplitude_eta_prime(amplitude_eta8, amplitude_eta1):
    theta_eta = np.arcsin(1/3)
    amplitude = amplitude_eta8*np.sin(theta_eta) + amplitude_eta1*np.cos(theta_eta)
    return amplitude

def amplitude_eta_eta(amplitude8X8,amplitude8X1,amplitude1X1):
    theta_eta = np.arcsin(1/3)
    amplitude = amplitude8X8*np.cos(theta_eta)**2 - 2*amplitude8X1*np.cos(theta_eta)*np.sin(theta_eta) + amplitude1X1*np.sin(theta_eta)**2
    return amplitude

def amplitude_eta_prime_eta_prime(amplitude8X8,amplitude8X1,amplitude1X1):
    theta_eta = np.arcsin(1/3)
    amplitude = amplitude8X8*np.sin(theta_eta)**2 + 2*amplitude8X1*np.cos(theta_eta)*np.sin(theta_eta) + amplitude1X1*np.cos(theta_eta)**2
    return amplitude

def amplitude_eta_eta_prime(amplitude8X8,amplitude8X1,amplitude1X1):
    theta_eta = np.arcsin(1/3)
    amplitude = amplitude8X8*np.cos(theta_eta)*np.sin(theta_eta) + amplitude8X1*(np.cos(theta_eta)**2 - np.sin(theta_eta)**2) - amplitude1X1*np.cos(theta_eta)*np.sin(theta_eta)
    return amplitude

def chi2(parameters):
    ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1 = parameters
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
    TT8X8 = abs(ampT8X8)
    CC8X8 = abs(ampC8X8)*np.exp(1j*delC8X8)
    PPuc8X8= abs(ampPuc8X8)*np.exp(1j*delPuc8X8)
    AA8X8 = abs(ampA8X8)*np.exp(1j*delA8X8)
    PAuc8X8 = abs(ampPAuc8X8)*np.exp(1j*delPAuc8X8)
    PPtc8X8 = abs(ampPtc8X8)*np.exp(1j*delPtc8X8)
    PAtc8X8 = abs(ampPAtc8X8)*np.exp(1j*delPAtc8X8)

    TT8X1 = abs(ampT8X1)*np.exp(1j*delT8X1)
    CC8X1 = abs(ampC8X1)*np.exp(1j*delC8X1)
    PPuc8X1= abs(ampPuc8X1)*np.exp(1j*delPuc8X1)
    PPtc8X1 = abs(ampPtc8X1)*np.exp(1j*delPtc8X1)

    CC1X1 = abs(ampC1X1)*np.exp(1j*delC1X1)
    PPtc1X1 = abs(ampPtc1X1)*np.exp(1j*delPtc1X1)
    
    #===Diagrams with weak phases===#
    #=== 8X8 Final states ===#
    #===Delta S = 0===#
    TTd8X8 = TT8X8*Vubd
    CCd8X8 = CC8X8*Vubd
    PPucd8X8 = PPuc8X8*Vubd
    AAd8X8 = AA8X8*Vubd
    PAucd8X8 = PAuc8X8*Vubd
    PPtcd8X8 = PPtc8X8*Vtbd
    PAtcd8X8 = PAtc8X8*Vtbd

    TTd8X1 = TT8X1*Vubd
    CCd8X1 = CC8X1*Vubd
    PPucd8X1 = PPuc8X1*Vubd
    PPtcd8X1 = PPtc8X1*Vtbd

    CC1X1 = CC1X1*Vubd
    PPtc1X1 = PPtc1X1*Vtbd

    #===EWP Diagrams===#
    PEWTd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWCd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWAd8X8 = -3/2.0 * (c9+c10)/(c1+c2)*AA8X8*Vtbd

    PEWTd8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)+(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbd
    PEWCd8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)-(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbd

    #===Delta S = 1===#
    TTs8X8 = TT8X8*Vubs
    CCs8X8 = CC8X8*Vubs
    PPucs8X8 = PPuc8X8*Vubs
    AAs8X8 = AA8X8*Vubs
    PAucs8X8 = PAuc8X8*Vubs
    PPtcs8X8 = PPtc8X8*Vtbs
    PAtcs8X8 = PAtc8X8*Vtbs

    TTs8X1 = TT8X1*Vubs
    CCs8X1 = CC8X1*Vubs
    PPucs8X1 = PPuc8X1*Vubs
    PPtcs8X1 = PPtc8X1*Vtbs

    CC1X1 = CC1X1*Vubs
    PPtc1X1 = PPtc1X1*Vtbs

    #===EWP Diagrams===#
    PEWTs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWCs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWAs8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbs

    PEWTs8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)+(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbs
    PEWCs8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)-(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbs

    #===Diagrams with conjugate weak phases===#
    TTdb8X8 = TT8X8*Vubdb
    CCdb8X8 = CC8X8*Vubdb
    PPucdb8X8 = PPuc8X8*Vubdb
    AAdb8X8 = AA8X8*Vubdb
    PAucdb8X8 = PAuc8X8*Vubdb
    PPtcdb8X8 = PPtc8X8*Vtbdb
    PAtcdb8X8 = PAtc8X8*Vtbdb

    TTdb8X1 = TT8X1*Vubdb
    CCdb8X1 = CC8X1*Vubdb
    PPucdb8X1 = PPuc8X1*Vubdb
    PPtcdb8X1 = PPtc8X1*Vtbdb

    CCb1X1 = CC1X1*Vubdb
    PPtcb1X1 = PPtc1X1*Vtbdb

    #===EWP Diagrams===#
    PEWTdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWCdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWAdb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbdb

    PEWTdb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)+(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbdb
    PEWCdb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)-(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbdb

    #===Delta S = 1===#
    TTsb8X8 = TT8X8*Vubsb
    CCsb8X8 = CC8X8*Vubsb
    PPucsb8X8 = PPuc8X8*Vubsb
    AAsb8X8 = AA8X8*Vubsb
    PAucsb8X8 = PAuc8X8*Vubsb
    PPtcsb8X8 = PPtc8X8*Vtbsb
    PAtcsb8X8 = PAtc8X8*Vtbsb

    TTsb8X1 = TT8X1*Vubsb
    CCsb8X1 = CC8X1*Vubsb
    PPucsb8X1 = PPuc8X1*Vubsb
    PPtcsb8X1 = PPtc8X1*Vtbsb

    CCb1X1 = CC1X1*Vubsb
    PPtcb1X1 = PPtc1X1*Vtbsb

    #===EWP Diagrams===#
    PEWTsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWCsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWAsb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbsb

    PEWTsb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)+(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbsb
    PEWCsb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)-(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbsb

    #===Diagram contributions to amplitudes===#

    DIAGd08X8 = [[0,0,1,1,0,1,0,0,-1/3,0],\
                [-1/root2,-1/root2,0,0,0,0,0,-1/root2,-1/root2,0],\
                [-1/root6,-1/root6,-2/root6,-2/root6,0,-2/root6,0,-1/root6,-1/(3*root6),0],\
                [0,0,1,0,1,1,1,0,-1/3,-2/3],\
                [-1,0,-1,0,-1,-1,-1,0,-2/3,-1/3],\
                [0,0,0,0,-1,0,-1,0,0,-1/3],\
                [0,-1/root2,1/root2,0,1/root2,1/root2,1/root2,-1/root2,-1/(3*root2),1/(3*root2)],\
                [0,0,-1/root3,0,0,-1/root3,0,0,1/(3*root3),1/root3],\
                [0,1/(3*root2),1/(3*root2),0,1/root2,1/(3*root2),1/root2,1/(3*root2),-1/(9*root2),-1/(3*root2)],\
                [-1,0,-1,0,0,-1,0,0,-2/3,0],\
                [0,-1/root2,1/root2,0,0,1/root2,0,-1/root2,-1/(3*root2),0],\
                [0,-1/root6,1/root6,0,0,1/root6,0,-1/root6,-1/(3*root6),0]]
    DIAGd18X8 = [[0,0,1,1,0,1,0,0,-1/3,0],\
                [-1/root2,-1/root2,-1/root2,-1/root2,0,-1/root2,0,-1/root2,-root2/3,0],\
                [-1/root6,-1/root6,1/root6,1/root6,0,1/root6,0,-1/root6,-4/(3*root6),0],\
                [-1,0,-1,0,0,-1,0,0,-2/3,0],\
                [0,-1/root2,1/root2,0,0,1/root2,0,-1/root2,-1/(3*root2),0],\
                [0,-1/root6,1/root6,0,0,1/root6,0,-1/root6,-1/(3*root6),0],\
                [0,0,1,0,1,1,1,0,-1/3,-2/3],\
                [0,0,0,0,-1,0,-1,0,0,-1/3],\
                [-1,0,-1,0,-1,-1,-1,0,-2/3,-1/3],\
                [0,0,0,0,1/root2,0,1/root2,0,0,1/(3*root2)],\
                [0,-1/root3,0,0,0,0,0,-1/root3,0,1/root3],\
                [0,-root2/3,2*root2/3,0,1/root2,2*root2/3,1/root2,-root2/3,-2*root2/9,-1/(3*root2)]]
    DIAGD08X1 = [[1/root3,1/root3,2/root3,2/root3,0,1/(3*root3)],\
                [0,0,2/root6,2/root6,-1/root6,-2/(3*root6)],\
                [-root2/3,-root2/3,-2/root3,-2/root3,-1/(3*root2),root2/9],\
                [0,1/root3,2/root3,2/root3,0,-2/(3*root3)]]
    DIAGD18X1 = [[1/root3,1/root3,2/root3,2/root3,0,1/(3*root3)],\
                [0,1/root3,2/root3,2/root3,0,-2/(3*root3)],\
                [0,-1/root6,0,0,-1/root6,0],\
                [0,1/(3*root2),2*root2/3,2*root2/3,-1/(3*root2),-2*root2/9]]
    DIAGD01X1 = [[root2/3,root2/3]]
    DIAGD11X1 = [[root2/3,root2/3]]


    #===Vector of diagrams===#
    diagramsd08X8 = np.array([TTd8X8,CCd8X8,PPucd8X8,AAd8X8,PAucd8X8,PPtcd8X8,PAtcd8X8,PEWTd8X8,PEWCd8X8,PEWAd8X8])
    diagramsd18X8 = np.array([TTs8X8,CCs8X8,PPucs8X8,AAs8X8,PAucs8X8,PPtcs8X8,PAtcs8X8,PEWTs8X8,PEWCs8X8,PEWAs8X8])
    diagramsd0b8X8= np.array([TTdb8X8,CCdb8X8,PPucdb8X8,AAdb8X8,PAucdb8X8,PPtcdb8X8,PAtcdb8X8,PEWTdb8X8,PEWCdb8X8,PEWAdb8X8])
    diagramsd1b8X8= np.array([TTsb8X8,CCsb8X8,PPucsb8X8,AAsb8X8,PAucsb8X8,PPtcsb8X8,PAtcsb8X8,PEWTsb8X8,PEWCsb8X8,PEWAsb8X8])
    
    diagramsd08X1 = np.array([TTd8X1,CCd8X1,PPucd8X1,PPtcd8X1,PEWTd8X1,PEWCd8X1])
    diagramsd18X1 = np.array([TTs8X1,CCs8X1,PPucs8X1,PPtcs8X1,PEWTs8X1,PEWCs8X1])
    diagramsd0b8X1= np.array([TTdb8X1,CCdb8X1,PPucdb8X1,PPtcdb8X1,PEWTdb8X1,PEWCdb8X1])
    diagramsd1b8X1= np.array([TTsb8X1,CCsb8X1,PPucsb8X1,PPtcsb8X1,PEWTsb8X1,PEWCsb8X1])

    diagramsd01X1 = np.array([CC1X1,PPtc1X1])
    diagramsd11X1 = np.array([CC1X1,PPtc1X1])
    diagramsd0b1X1= np.array([CCb1X1,PPtcb1X1])
    diagramsd1b1X1= np.array([CCb1X1,PPtcb1X1])

    #===Amplitudes===#
    #===Initialising amplitudes===#
    amplitudes_DeltaS08X8  = np.zeros(12)
    amplitudes_DeltaS18X8  = np.zeros(12)
    amplitudes_DeltaS0b8X8 = np.zeros(12)
    amplitudes_DeltaS1b8X8 = np.zeros(12)

    amplitudes_DeltaS08X1 = np.zeros(6)
    amplitudes_DeltaS18X1 = np.zeros(6)
    amplitudes_DeltaS0b8X1= np.zeros(6)
    amplitudes_DeltaS1b8X1= np.zeros(6)

    amplitudes_DeltaS01X1 = np.zeros(2)
    amplitudes_DeltaS11X1 = np.zeros(2)
    amplitudes_DeltaS0b1X1= np.zeros(2)
    amplitudes_DeltaS1b1X1= np.zeros(2)

    #===Calculating amplitudes===#
    amplitudes_DeltaS08X8 = np.matmul(DIAGd08X8,diagramsd08X8)
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS0b8X8= np.matmul(DIAGd08X8,diagramsd0b8X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    amplitudes_DeltaS08X1 = np.matmul(DIAGD08X1,diagramsd08X1)
    amplitudes_DeltaS18X1 = np.matmul(DIAGD18X1,diagramsd18X1)
    amplitudes_DeltaS0b8X1= np.matmul(DIAGD08X1,diagramsd0b8X1)
    amplitudes_DeltaS1b8X1= np.matmul(DIAGD18X1,diagramsd1b8X1)

    amplitudes_DeltaS01X1 = np.matmul(DIAGD01X1,diagramsd01X1)
    amplitudes_DeltaS11X1 = np.matmul(DIAGD11X1,diagramsd11X1)
    amplitudes_DeltaS0b1X1= np.matmul(DIAGD01X1,diagramsd0b1X1)
    amplitudes_DeltaS1b1X1= np.matmul(DIAGD11X1,diagramsd1b1X1)

    #===Calculating branching ratios===#
    #===Branching ratios Delta S = 0===#
    BR_BpK0bKp = (fBpK0bKp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[0])**2 + abs(amplitudes_DeltaS0b8X8[0])**2)/GammaBp
    BR_BpP0Pp  = (fBpP0Pp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[1])**2 + abs(amplitudes_DeltaS0b8X8[1])**2)/GammaBp
    BR_BpEPp   = (fBpEPp/(16*math.pi*mBp**2))*(abs(amplitude_eta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2 + abs(amplitude_eta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2)/GammaBp
    BR_BpEpPp  = (fBpEpPp/(16*math.pi*mBp**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2)/GammaBp
    BR_B0K0K0b = (fB0K0K0b/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[3])**2 +abs(amplitudes_DeltaS0b8X8[3])**2)/GammaB0
    BR_B0PpPm  = (fB0PpPm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[4])**2 + abs(amplitudes_DeltaS0b8X8[4])**2)/GammaB0
    BR_B0KpKm  = (fB0KpKm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[5])**2 + abs(amplitudes_DeltaS0b8X8[5])**2)/GammaB0
    BR_B0P0P0  = (fB0P0P0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[6])**2 + abs(amplitudes_DeltaS0b8X8[6])**2)/GammaB0
    BR_B0P0E   = (fB0P0E/(16*math.pi*mB0**2))*(abs(amplitude_eta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2 + abs(amplitude_eta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2)/GammaB0
    BR_B0P0Ep  = (fB0P0Ep/(16*math.pi*mB0**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2)/GammaB0
    BR_B0sPpKm = (fB0sPpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[9])**2 + abs(amplitudes_DeltaS0b8X8[9])**2)/GammaB0s
    BR_B0sP0K0b= (fB0sP0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[10])**2 + abs(amplitudes_DeltaS0b8X8[10])**2)/GammaB0s
    BR_B0sEK0b = (fB0sEK0b/(16*math.pi*mB0s**2))*(abs(amplitude_eta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2 + abs(amplitude_eta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2)/GammaB0s
    BR_B0sEpK0b= (fB0sEpK0b/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2)/GammaB0s
    
    BR_B0EE    = (fB0EE/(16*math.pi*mB0**2))*((abs(amplitude_eta_eta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0])))**2 + abs(amplitude_eta_eta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2)/GammaB0
    BR_B0EEp   = (fB0EEp/(16*math.pi*mB0**2))*((abs(amplitude_eta_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0])))**2 + abs(amplitude_eta_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2)/GammaB0
    BR_B0EpEp  = (fB0EpEp/(16*math.pi*mB0**2))*((abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0])))**2 + abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2)/GammaB0

    #===Branching ratios Delta S = 1===#
    BR_BpPpK0  = (fBpPpK0/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[0])**2 + abs(amplitudes_DeltaS1b8X8[0])**2)/GammaBp
    BR_BpP0Kp  = (fBpP0Kp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[1])**2 + abs(amplitudes_DeltaS1b8X8[1])**2)/GammaBp
    BR_BpEKp   = (fBpEKp/(16*math.pi*mBp**2))*(abs(amplitude_eta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2 + abs(amplitude_eta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2)/GammaBp
    BR_BpEpKp  = (fBpEpKp/(16*math.pi*mBp**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2)/GammaBp
    BR_B0PmKp  = (fB0PmKp/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[3])**2 + abs(amplitudes_DeltaS1b8X8[3])**2)/GammaB0
    BR_B0P0K0  = (fB0P0K0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[4])**2 + abs(amplitudes_DeltaS1b8X8[4])**2)/GammaB0
    BR_B0EK0   = (fB0EK0/(16*math.pi*mB0**2))*(abs(amplitude_eta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2 + abs(amplitude_eta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2)/GammaB0
    BR_B0EpK0  = (fB0EpK0/(16*math.pi*mB0**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2)/GammaB0
    BR_B0sK0K0b= (fB0sK0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[6])**2 + abs(amplitudes_DeltaS1b8X8[6])**2)/GammaB0s
    BR_B0sPpPm = (fB0sPpPm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[7])**2 + abs(amplitudes_DeltaS1b8X8[7])**2)/GammaB0s
    BR_B0sKpKm = (fB0sKpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[8])**2 + abs(amplitudes_DeltaS1b8X8[8])**2)/GammaB0s
    BR_B0sP0P0 = (fB0sP0P0/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[9])**2 + abs(amplitudes_DeltaS1b8X8[9])**2)/GammaB0s
    BR_B0sP0E  = (fB0sP0E/(16*math.pi*mB0s**2))*(abs(amplitude_eta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2 + abs(amplitude_eta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2)/GammaB0s
    BR_B0sP0Ep = (fB0sP0Ep/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2)/GammaB0s

    BR_B0sEE   = (fB0sEE/(16*math.pi*mB0s**2))*(abs(amplitude_eta_eta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2 + abs(amplitude_eta_eta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2)/GammaB0s
    BR_B0sEEp  = (fB0sEEp/(16*math.pi*mB0s**2))*(abs(amplitude_eta_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2 + abs(amplitude_eta_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2)/GammaB0s
    BR_B0sEpEp = (fB0sEpEp/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2 + abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2)/GammaB0s

    #===Direct CP asymmetries Delta S = 0===#
    ACP_BpK0bKp = (abs(amplitudes_DeltaS0b8X8[0])**2 - abs(amplitudes_DeltaS08X8[0])**2)/(abs(amplitudes_DeltaS0b8X8[0])**2 + abs(amplitudes_DeltaS08X8[0])**2)
    ACP_BpP0Pp  = (abs(amplitudes_DeltaS0b8X8[1])**2 - abs(amplitudes_DeltaS08X8[1])**2)/(abs(amplitudes_DeltaS0b8X8[1])**2 + abs(amplitudes_DeltaS08X8[1])**2)
    ACP_BpEPp   = (abs(amplitude_eta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2 - abs(amplitude_eta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2)/(abs(amplitude_eta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2 + abs(amplitude_eta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2)
    ACP_BpEpPp  = (abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0]))**2)
    ACP_B0K0K0b = (abs(amplitudes_DeltaS0b8X8[3])**2 - abs(amplitudes_DeltaS08X8[3]**2))/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    ACP_B0PpPm  = (abs(amplitudes_DeltaS0b8X8[4])**2 - abs(amplitudes_DeltaS08X8[4])**2)/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    ACP_B0KpKm  = (abs(amplitudes_DeltaS0b8X8[5])**2 - abs(amplitudes_DeltaS08X8[5])**2)/(abs(amplitudes_DeltaS0b8X8[5])**2 + abs(amplitudes_DeltaS08X8[5])**2)
    ACP_B0P0P0  = (abs(amplitudes_DeltaS0b8X8[6])**2 - abs(amplitudes_DeltaS08X8[6])**2)/(abs(amplitudes_DeltaS0b8X8[6])**2 + abs(amplitudes_DeltaS08X8[6])**2)
    ACP_B0P0E   = (abs(amplitude_eta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2 - abs(amplitude_eta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2)/(abs(amplitude_eta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2 + abs(amplitude_eta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2)
    ACP_B0P0Ep  = (abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1]))**2)
    ACP_B0sPpKm = (abs(amplitudes_DeltaS0b8X8[9])**2 - abs(amplitudes_DeltaS08X8[9])**2)/(abs(amplitudes_DeltaS0b8X8[9])**2 + abs(amplitudes_DeltaS08X8[9])**2)
    ACP_B0sP0K0b= (abs(amplitudes_DeltaS0b8X8[10])**2 - abs(amplitudes_DeltaS08X8[10])**2)/(abs(amplitudes_DeltaS0b8X8[10])**2 + abs(amplitudes_DeltaS08X8[10])**2)
    ACP_B0sEK0b = (abs(amplitude_eta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2 - abs(amplitude_eta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2)/(abs(amplitude_eta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2 + abs(amplitude_eta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2)
    ACP_B0sEpK0b= (abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3]))**2)

    ACP_B0EE    = (abs(amplitude_eta_eta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 - abs(amplitude_eta_eta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)/(abs(amplitude_eta_eta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 + abs(amplitude_eta_eta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)
    ACP_B0EEp   = (abs(amplitude_eta_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 - abs(amplitude_eta_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)/(abs(amplitude_eta_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 + abs(amplitude_eta_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)
    ACP_B0EpEp  = (abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 - abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)/(abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0]))**2 + abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0]))**2)
    
    #===Direct CP asymmetries Delta S = 1===#
    ACP_BpPpK0  = (abs(amplitudes_DeltaS1b8X8[0])**2 - abs(amplitudes_DeltaS18X8[0])**2)/(abs(amplitudes_DeltaS1b8X8[0])**2 + abs(amplitudes_DeltaS18X8[0])**2)
    ACP_BpP0Kp  = (abs(amplitudes_DeltaS1b8X8[1])**2 - abs(amplitudes_DeltaS18X8[1])**2)/(abs(amplitudes_DeltaS1b8X8[1])**2 + abs(amplitudes_DeltaS18X8[1])**2)
    ACP_BpEKp   = (abs(amplitude_eta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2 - abs(amplitude_eta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2)/(abs(amplitude_eta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2 + abs(amplitude_eta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2)
    ACP_BpEpKp  = (abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0]))**2)
    ACP_B0PmKp  = (abs(amplitudes_DeltaS1b8X8[3])**2 - abs(amplitudes_DeltaS18X8[3])**2)/(abs(amplitudes_DeltaS1b8X8[3])**2 + abs(amplitudes_DeltaS18X8[3])**2)
    ACP_B0P0K0  = (abs(amplitudes_DeltaS1b8X8[4])**2 - abs(amplitudes_DeltaS18X8[4])**2)/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    ACP_B0EK0   = (abs(amplitude_eta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 - abs(amplitude_eta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)/(abs(amplitude_eta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
    ACP_B0EpK0  = (abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
    ACP_B0sK0K0b= (abs(amplitudes_DeltaS1b8X8[6])**2 - abs(amplitudes_DeltaS18X8[6])**2)/(abs(amplitudes_DeltaS1b8X8[6])**2 + abs(amplitudes_DeltaS18X8[6])**2)
    ACP_B0sPpPm = (abs(amplitudes_DeltaS1b8X8[7])**2 - abs(amplitudes_DeltaS18X8[7])**2)/(abs(amplitudes_DeltaS1b8X8[7])**2 + abs(amplitudes_DeltaS18X8[7])**2)
    ACP_B0sKpKm = (abs(amplitudes_DeltaS1b8X8[8])**2 - abs(amplitudes_DeltaS18X8[8])**2)/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    ACP_B0sP0P0 = (abs(amplitudes_DeltaS1b8X8[9])**2 - abs(amplitudes_DeltaS18X8[9])**2)/(abs(amplitudes_DeltaS1b8X8[9])**2 + abs(amplitudes_DeltaS18X8[9])**2)
    ACP_B0sP0E  = (abs(amplitude_eta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2 - abs(amplitude_eta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2)/(abs(amplitude_eta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2 + abs(amplitude_eta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2)
    ACP_B0sP0Ep = (abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2 - abs(amplitude_eta_prime(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2)/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2]))**2)

    ACP_B0sEE   = (abs(amplitude_eta_eta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 - abs(amplitude_eta_eta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)/(abs(amplitude_eta_eta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 + abs(amplitude_eta_eta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)
    ACP_B0sEEp  = (abs(amplitude_eta_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 - abs(amplitude_eta_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)/(abs(amplitude_eta_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 + abs(amplitude_eta_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)
    ACP_B0sEpEp = (abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 - abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)/(abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0]))**2 + abs(amplitude_eta_prime_eta_prime(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0]))**2)

    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS0b8X8[4])*amplitudes_DeltaS08X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*betas)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))*amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
    Im_SCP_B0EpK0  = SCP_B0EpK0.imag

    #===Chi2 Contributions===#
    #===Branching ratios===#
    chi2_BR_BpK0bKp = (BR_BpK0bKp - BpK0bKp_exp)**2/(BpK0bKp_inc**2)
    chi2_BR_BpP0Pp  = (BR_BpP0Pp - BpP0Pp_exp)**2/(BpP0Pp_inc**2)
    chi2_BR_BpEPp   = (BR_BpEPp - BpEPp_exp)**2/(BpEPp_inc**2)
    chi2_BR_BpEpPp  = (BR_BpEpPp - BpEpPp_exp)**2/(BpEpPp_inc**2)
    chi2_BR_B0K0K0b = (BR_B0K0K0b - B0K0K0b_exp)**2/(B0K0K0b_inc**2)
    chi2_BR_B0PpPm  = (BR_B0PpPm - B0PpPm_exp)**2/(B0PpPm_inc**2)
    chi2_BR_B0KpKm  = (BR_B0KpKm - B0KpKm_exp)**2/(B0KpKm_inc**2)
    chi2_BR_B0P0P0  = (BR_B0P0P0 - B0P0P0_exp)**2/(B0P0P0_inc**2)
    chi2_BR_B0P0E   = (BR_B0P0E - B0P0E_exp)**2/(B0P0E_inc**2)
    chi2_BR_B0P0Ep  = (BR_B0P0Ep - B0P0Ep_exp)**2/(B0P0Ep_inc**2)
    chi2_BR_B0EE    = (BR_B0EE - B0EE_exp)**2/(B0EE_inc**2)
    #chi2_BR_B0EEp   = (BR_B0EEp - B0EEp_exp)**2/(B0EEp_inc**2)
    chi2_BR_B0EpEp  = (BR_B0EpEp - B0EpEp_exp)**2/(B0EpEp_inc**2)
    chi2_BR_B0sPpKm = (BR_B0sPpKm - B0sPpKm_exp)**2/(B0sPpKm_inc**2)
    #chi2_BR_B0sP0K0b= (BR_B0sP0K0b - B0sP0K0b_exp)**2/(B0sP0K0b_inc**2)
    #chi2_BR_B0sEK0b = (BR_B0sEK0b - B0sEK0b_exp)**2/(B0sEK0b_inc**2)
    #chi2_BR_B0sEpK0b= (BR_B0sEpK0b - B0sEpK0b_exp)**2/(B0sEpK0b_inc**2)

    chi2_BR_BpPpK0  = (BR_BpPpK0 - BpPpK0_exp)**2/(BpPpK0_inc**2)
    chi2_BR_BpP0Kp  = (BR_BpP0Kp - BpP0Kp_exp)**2/(BpP0Kp_inc**2)
    chi2_BR_BpEKp   = (BR_BpEKp - BpEKp_exp)**2/(BpEKp_inc**2)
    chi2_BR_BpEpKp  = (BR_BpEpKp - BpEpKp_exp)**2/(BpEpKp_inc**2)
    chi2_BR_B0PmKp  = (BR_B0PmKp - B0PmKp_exp)**2/(B0PmKp_inc**2)
    chi2_BR_B0P0K0  = (BR_B0P0K0 - B0P0K0_exp)**2/(B0P0K0_inc**2)
    chi2_BR_B0EK0   = (BR_B0EK0 - B0EK0_exp)**2/(B0EK0_inc**2)
    chi2_BR_B0EpK0  = (BR_B0EpK0 - B0EpK0_exp)**2/(B0EpK0_inc**2)
    chi2_BR_B0sK0K0b= (BR_B0sK0K0b - B0sK0K0b_exp)**2/(B0sK0K0b_inc**2)
    chi2_BR_B0sPpPm = (BR_B0sPpPm - B0sPpPm_exp)**2/(B0sPpPm_inc**2)
    chi2_BR_B0sKpKm = (BR_B0sKpKm - B0sKpKm_exp)**2/(B0sKpKm_inc**2)
    chi2_BR_B0sP0P0 = (BR_B0sP0P0 - B0sP0P0_exp)**2/(B0sP0P0_inc**2)
    #chi2_BR_B0sP0E  = (BR_B0sP0E - B0sP0E_exp)**2/(B0sP0E_inc**2)
    #chi2_BR_B0sP0Ep = (BR_B0sP0Ep - B0sP0Ep_exp)**2/(B0sP0Ep_inc**2)
    chi2_BR_B0sEE   = (BR_B0sEE - B0sEE_exp)**2/(B0sEE_inc**2)
    chi2_BR_B0sEEp  = (BR_B0sEEp - B0sEEp_exp)**2/(B0sEEp_inc**2)
    chi2_BR_B0sEpEp = (BR_B0sEpEp - B0sEpEp_exp)**2/(B0sEpEp_inc**2)

    #===Direct CP asymmetries===#
    chi2_ACP_BpK0bKp = (ACP_BpK0bKp - ACP_BpK0bKp_exp)**2/(ACP_BpK0bKp_inc**2)
    chi2_ACP_BpP0Pp  = (ACP_BpP0Pp - ACP_BpP0Pp_exp)**2/(ACP_BpP0Pp_inc**2)
    chi2_ACP_BpEPp   = (ACP_BpEPp - ACP_BpEPp_exp)**2/(ACP_BpEPp_inc**2)
    chi2_ACP_BpEpPp  = (ACP_BpEpPp - ACP_BpEpPp_exp)**2/(ACP_BpEpPp_inc**2)
    chi2_ACP_B0K0K0b = (ACP_B0K0K0b - ACP_B0K0K0b_exp)**2/(ACP_B0K0K0b_inc**2)
    chi2_ACP_B0PpPm  = (ACP_B0PpPm - ACP_B0PpPm_exp)**2/(ACP_B0PpPm_inc**2)
    #chi2_ACP_B0KpKm  = (ACP_B0KpKm - ACP_B0KpKm_exp)**2/(ACP_B0KpKm_inc**2)
    chi2_ACP_B0P0P0  = (ACP_B0P0P0 - ACP_B0P0P0_exp)**2/(ACP_B0P0P0_inc**2)
    #chi2_ACP_B0P0E   = (ACP_B0P0E - ACP_B0P0E_exp)**2/(ACP_B0P0E_inc**2)
    #chi2_ACP_B0P0Ep  = (ACP_B0P0Ep - ACP_B0P0Ep_exp)**2/(ACP_B0P0Ep_inc**2)
    #chi2_ACP_B0EE    = (ACP_B0EE - ACP_B0EE_exp)**2/(ACP_B0EE_inc**2)
    #chi2_ACP_B0EEp   = (ACP_B0EEp - ACP_B0EEp_exp)**2/(ACP_B0EEp_inc**2)
    #chi2_ACP_B0EpEp  = (ACP_B0EpEp - ACP_B0EpEp_exp)**2/(ACP_B0EpEp_inc**2)
    chi2_ACP_B0sPpKm = (ACP_B0sPpKm - ACP_B0sPpKm_exp)**2/(ACP_B0sPpKm_inc**2)
    #chi2_ACP_B0sP0K0b= (ACP_B0sP0K0b - ACP_B0sP0K0b_exp)**2/(ACP_B0sP0K0b_inc**2)
    #chi2_ACP_B0sEK0b = (ACP_B0sEK0b - ACP_B0sEK0b_exp)**2/(ACP_B0sEK0b_inc**2)
    #chi2_ACP_B0sEpK0b= (ACP_B0sEpK0b - ACP_B0sEpK0b_exp)**2/(ACP_B0sEpK0b_inc**2)

    chi2_ACP_BpPpK0  = (ACP_BpPpK0 - ACP_BpPpK0_exp)**2/(ACP_BpPpK0_inc**2)
    chi2_ACP_BpP0Kp  = (ACP_BpP0Kp - ACP_BpP0Kp_exp)**2/(ACP_BpP0Kp_inc**2)
    chi2_ACP_BpEKp   = (ACP_BpEKp - ACP_BpEKp_exp)**2/(ACP_BpEKp_inc**2)
    chi2_ACP_BpEpKp  = (ACP_BpEpKp - ACP_BpEpKp_exp)**2/(ACP_BpEpKp_inc**2)
    chi2_ACP_B0PmKp  = (ACP_B0PmKp - ACP_B0PmKp_exp)**2/(ACP_B0PmKp_inc**2)
    chi2_ACP_B0P0K0  = (ACP_B0P0K0 - ACP_B0P0K0_exp)**2/(ACP_B0P0K0_inc**2)
    #chi2_ACP_B0EK0   = (ACP_B0EK0 - ACP_B0EK0_exp)**2/(ACP_B0EK0_inc**2)
    chi2_ACP_B0EpK0  = (ACP_B0EpK0 - ACP_B0EpK0_exp)**2/(ACP_B0EpK0_inc**2)
    #chi2_ACP_B0sK0K0b= (ACP_B0sK0K0b - ACP_B0sK0K0b_exp)**2/(ACP_B0sK0K0b_inc**2)
    #chi2_ACP_B0sPpPm = (ACP_B0sPpPm - ACP_B0sPpPm_exp)**2/(ACP_B0sPpPm_inc**2)
    chi2_ACP_B0sKpKm = (ACP_B0sKpKm - ACP_B0sKpKm_exp)**2/(ACP_B0sKpKm_inc**2)
    #chi2_ACP_B0sP0P0 = (ACP_B0sP0P0 - ACP_B0sP0P0_exp)**2/(ACP_B0sP0P0_inc**2)
    #chi2_ACP_B0sP0E  = (ACP_B0sP0E - ACP_B0sP0E_exp)**2/(ACP_B0sP0E_inc**2)
    #chi2_ACP_B0sP0Ep = (ACP_B0sP0Ep - ACP_B0sP0Ep_exp)**2/(ACP_B0sP0Ep_inc**2)
    #chi2_ACP_B0sEE   = (ACP_B0sEE - ACP_B0sEE_exp)**2/(ACP_B0sEE_inc**2)
    #chi2_ACP_B0sEEp  = (ACP_B0sEEp - ACP_B0sEEp_exp)**2/(ACP_B0sEEp_inc**2)
    #chi2_ACP_B0sEpEp = (ACP_B0sEpEp - ACP_B0sEpEp_exp)**2/(ACP_B0sEpEp_inc**2)

    #===Indirect CP asymmetries===#
    chi2_SCP_B0K0K0b    = (Im_SCP_B0K0K0b - SCP_B0K0K0b_exp)**2/(SCP_B0K0K0b_inc**2)
    chi2_SCP_B0PpPm     = (Im_SCP_B0PpPm - SCP_B0PpPm_exp)**2/(SCP_B0PpPm_inc**2)

    chi2_SCP_B0P0K0     = (Im_SCP_B0P0K0 - SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc**2)
    chi2_SCP_B0sKpKm    = (Im_SCP_B0sKpKm - SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc**2)
    chi2_SCP_B0EpK0     = (Im_SCP_B0EpK0 - SCP_B0EpK0_exp)**2/(SCP_B0EpK0_inc**2)

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_BpEPp + chi2_BR_BpEpPp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0P0E + chi2_BR_B0P0Ep + chi2_BR_B0EE + chi2_BR_B0EpEp + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_BpEKp + chi2_BR_BpEpKp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0EK0 + chi2_BR_B0EpK0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0 + chi2_BR_B0sEE + chi2_BR_B0sEEp + chi2_BR_B0sEpEp
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_BpEPp + chi2_ACP_BpEpPp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_BpEKp + chi2_ACP_BpEpKp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0EpK0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm + chi2_SCP_B0EpK0

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return -chi2_total