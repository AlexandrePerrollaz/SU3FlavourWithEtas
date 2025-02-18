import numpy as np
import matplotlib.pyplot as plt
import math
from exp_data import *



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

# Amplitudes 8X8 8X1 and 1X1 with theta as a free parameter

def amplitude_eta_with_theta(amplitude_eta8, amplitude_eta1,theta_eta):
    amplitude = amplitude_eta8*np.cos(theta_eta) - amplitude_eta1*np.sin(theta_eta)
    return amplitude

def amplitude_eta_prime_with_theta(amplitude_eta8, amplitude_eta1,theta_eta):
    amplitude = amplitude_eta8*np.sin(theta_eta) + amplitude_eta1*np.cos(theta_eta)
    return amplitude

def amplitude_eta_eta_with_theta(amplitude8X8,amplitude8X1,amplitude1X1,theta_eta):
    amplitude = amplitude8X8*np.cos(theta_eta)**2 - 2*amplitude8X1*np.cos(theta_eta)*np.sin(theta_eta) + amplitude1X1*np.sin(theta_eta)**2
    return amplitude

def amplitude_eta_prime_eta_prime_with_theta(amplitude8X8,amplitude8X1,amplitude1X1,theta_eta):
    amplitude = amplitude8X8*np.sin(theta_eta)**2 + 2*amplitude8X1*np.cos(theta_eta)*np.sin(theta_eta) + amplitude1X1*np.cos(theta_eta)**2
    return amplitude

def amplitude_eta_eta_prime_with_theta(amplitude8X8,amplitude8X1,amplitude1X1,theta_eta):
    amplitude = amplitude8X8*np.cos(theta_eta)*np.sin(theta_eta) + amplitude8X1*(np.cos(theta_eta)**2 - np.sin(theta_eta)**2) - amplitude1X1*np.cos(theta_eta)*np.sin(theta_eta)
    return amplitude

def chi2(parameters):
    ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1 = parameters
    
    
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

def chi2WithoutEtaEta(parameters):
    ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1 = parameters
    
    
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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

    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    #===Indirect CP asymmetries===#
    chi2_SCP_B0K0K0b    = (Im_SCP_B0K0K0b - SCP_B0K0K0b_exp)**2/(SCP_B0K0K0b_inc**2)
    chi2_SCP_B0PpPm     = (Im_SCP_B0PpPm - SCP_B0PpPm_exp)**2/(SCP_B0PpPm_inc**2)

    chi2_SCP_B0P0K0     = (Im_SCP_B0P0K0 - SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc**2)
    chi2_SCP_B0sKpKm    = (Im_SCP_B0sKpKm - SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc**2)
    chi2_SCP_B0EpK0     = (Im_SCP_B0EpK0 - SCP_B0EpK0_exp)**2/(SCP_B0EpK0_inc**2)

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_BpEPp + chi2_BR_BpEpPp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0P0E + chi2_BR_B0P0Ep + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_BpEKp + chi2_BR_BpEpKp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0EK0 + chi2_BR_B0EpK0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_BpEPp + chi2_ACP_BpEpPp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_BpEKp + chi2_ACP_BpEpKp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0EpK0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm + chi2_SCP_B0EpK0

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return -chi2_total
#### Old version of the chi2 function ####

def oldchi2(ampT,ampC,ampP,ampA,ampPA,ampP2,ampPA2,delC,delP,delA,delPA,delP2,delPA2):

    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp
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
    T = abs(ampT)
    C = abs(ampC)*np.exp(1j*delC)
    P = abs(ampP)*np.exp(1j*delP)
    A = abs(ampA)*np.exp(1j*delA)
    PA = abs(ampPA)*np.exp(1j*delPA)
    P2 = abs(ampP2)*np.exp(1j*delP2)
    PA2 = abs(ampPA2)*np.exp(1j*delPA2)
    
    #Delta s = 0
    Td = T*Vubd
    Cd = C*Vubd
    Pd = P*Vubd
    Ad = A*Vubd
    PAd = PA*Vubd

    P2d = P2*Vtbd
    PA2d = PA2*Vtbd
    
    #EWP
    PEWd  = 3/4*(c9-c10)/(c1-c2)*(T-C-A)*Vtbd + 3/4*(c9+c10)/(c1+c2)*(C+T+A)*Vtbd
    PEWcd = 3/4*(c9-c10)/(c1-c2)*(A+C-T)*Vtbd + 3/4*(c9+c10)/(c1+c2)*(C+T-A)*Vtbd

    #Delta s = 1
    Ts = T*Vubs
    Cs = C*Vubs
    Ps = P*Vubs
    As = A*Vubs
    PAs = PA*Vubs

    P2s = P2*Vtbs
    PA2s = PA2*Vtbs

    #EWP
    PEWs = 3/4*(c9-c10)/(c1-c2)*(T-C-A)*Vtbs + 3/4*(c9+c10)/(c1+c2)*(C+T+A)*Vtbs
    PEWcs= 3/4*(c9-c10)/(c1-c2)*(A+C-T)*Vtbs + 3/4*(c9+c10)/(c1+c2)*(C+T-A)*Vtbs

    #Conjugate Delta s = 0
    Tdb = T*Vubdb
    Cdb = C*Vubdb
    Pdb = P*Vubdb
    Adb = A*Vubdb
    PAdb = PA*Vubdb

    P2db = P2*Vtbdb
    PA2db = PA2*Vtbdb

    #EWP
    PEWdb  = 3/4*(c9-c10)/(c1-c2)*(T-C-A)*Vtbdb + 3/4*(c9+c10)/(c1+c2)*(C+T+A)*Vtbdb
    PEWcdb = 3/4*(c9-c10)/(c1-c2)*(A+C-T)*Vtbdb + 3/4*(c9+c10)/(c1+c2)*(C+T-A)*Vtbdb

    #Conjugate Delta s = 1
    Tsb = T*Vubsb
    Csb = C*Vubsb
    Psb = P*Vubsb
    Asb = A*Vubsb
    PAsb = PA*Vubsb

    P2sb = P2*Vtbsb
    PA2sb = PA2*Vtbsb

    #EWP
    PEWsb = 3/4*(c9-c10)/(c1-c2)*(T-C-A)*Vtbsb + 3/4*(c9+c10)/(c1+c2)*(C+T+A)*Vtbsb
    PEWcsb= 3/4*(c9-c10)/(c1-c2)*(A+C-T)*Vtbsb + 3/4*(c9+c10)/(c1+c2)*(C+T-A)*Vtbsb

    #========== RME matrix ==========#

    DIAGd0 = np.array([[0,0,1,0,1,1,1],[-1,0,-1,0,-1,-1,-1],\
                       [0,0,0,0,-1,0,-1],[0,-1/np.sqrt(2),1/np.sqrt(2),0,1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)],\
                       [-1,0,-1,0,0,-1,0],[0,-1/np.sqrt(2),1/np.sqrt(2),0,0,1/np.sqrt(2),0],\
                       [0,0,1,1,0,1,0],[-1/np.sqrt(2),-1/np.sqrt(2),0,0,0,0,0]])
    
    EWPd0  = np.array([[0,1/3],[0,2/3],[0,0],[1/np.sqrt(2),1/(3*np.sqrt(2))],[0,2/3],[1/np.sqrt(2),1/(3*np.sqrt(2))],[0,1/3],[1/np.sqrt(2),1/np.sqrt(2)]])
    
    DIAGd1 = np.array([[-1,0,-1,0,0,-1,0],[0,-1/np.sqrt(2),1/np.sqrt(2),0,0,1/np.sqrt(2),0],\
                       [0,0,1,0,1,1,1],[0,0,0,0,-1,0,-1],\
                       [-1,0,-1,0,-1,-1,-1],[0,0,0,0,1/np.sqrt(2),0,1/np.sqrt(2)],\
                       [0,0,1,1,0,1,0],[-1/np.sqrt(2),-1/np.sqrt(2),-1/np.sqrt(2),-1/np.sqrt(2),0,-1/np.sqrt(2),0]])
    
    EWPd1  = np.array([[0,2/3],[1/np.sqrt(2),1/(3*np.sqrt(2))],[0,1/3],[0,0],[0,2/3],[0,0],[0,1/3],[1/np.sqrt(2),np.sqrt(2)/3]])
    
    #===Amplitudes in terms of effective diagrams===#
    AB0K0K0b = Td*DIAGd0[0,0] + Cd*DIAGd0[0,1] + Pd*DIAGd0[0,2] + Ad*DIAGd0[0,3] + PAd*DIAGd0[0,4] + P2d*DIAGd0[0,5] + PA2d*DIAGd0[0,6] + PEWd*EWPd0[0,0] + PEWcd*EWPd0[0,1]
    print(AB0K0K0b)
    AB0PmPp  = Td*DIAGd0[1,0] + Cd*DIAGd0[1,1] + Pd*DIAGd0[1,2] + Ad*DIAGd0[1,3] + PAd*DIAGd0[1,4] + P2d*DIAGd0[1,5] + PA2d*DIAGd0[1,6] + PEWd*EWPd0[1,0] + PEWcd*EWPd0[1,1]
    AB0KmKp  = Td*DIAGd0[2,0] + Cd*DIAGd0[2,1] + Pd*DIAGd0[2,2] + Ad*DIAGd0[2,3] + PAd*DIAGd0[2,4] + P2d*DIAGd0[2,5] + PA2d*DIAGd0[2,6] + PEWd*EWPd0[2,0] + PEWcd*EWPd0[2,1]
    AB0P0P0  = Td*DIAGd0[3,0] + Cd*DIAGd0[3,1] + Pd*DIAGd0[3,2] + Ad*DIAGd0[3,3] + PAd*DIAGd0[3,4] + P2d*DIAGd0[3,5] + PA2d*DIAGd0[3,6] + PEWd*EWPd0[3,0] + PEWcd*EWPd0[3,1]
    AB0sPpKm = Td*DIAGd0[4,0] + Cd*DIAGd0[4,1] + Pd*DIAGd0[4,2] + Ad*DIAGd0[4,3] + PAd*DIAGd0[4,4] + P2d*DIAGd0[4,5] + PA2d*DIAGd0[4,6] + PEWd*EWPd0[4,0] + PEWcd*EWPd0[4,1]
    AB0sP0K0b= Td*DIAGd0[5,0] + Cd*DIAGd0[5,1] + Pd*DIAGd0[5,2] + Ad*DIAGd0[5,3] + PAd*DIAGd0[5,4] + P2d*DIAGd0[5,5] + PA2d*DIAGd0[5,6] + PEWd*EWPd0[5,0] + PEWcd*EWPd0[5,1]
    ABpK0bKp = Td*DIAGd0[6,0] + Cd*DIAGd0[6,1] + Pd*DIAGd0[6,2] + Ad*DIAGd0[6,3] + PAd*DIAGd0[6,4] + P2d*DIAGd0[6,5] + PA2d*DIAGd0[6,6] + PEWd*EWPd0[6,0] + PEWcd*EWPd0[6,1]
    ABpP0Pp  = Td*DIAGd0[7,0] + Cd*DIAGd0[7,1] + Pd*DIAGd0[7,2] + Ad*DIAGd0[7,3] + PAd*DIAGd0[7,4] + P2d*DIAGd0[7,5] + PA2d*DIAGd0[7,6] + PEWd*EWPd0[7,0] + PEWcd*EWPd0[7,1]
    
    AB0PmKp  = Ts*DIAGd1[0,0] + Cs*DIAGd1[0,1] + Ps*DIAGd1[0,2] + As*DIAGd1[0,3] + PAs*DIAGd1[0,4] + P2s*DIAGd1[0,5] + PA2s*DIAGd1[0,6] + PEWs*EWPd1[0,0] + PEWcs*EWPd1[0,1]
    AB0P0K0  = Ts*DIAGd1[1,0] + Cs*DIAGd1[1,1] + Ps*DIAGd1[1,2] + As*DIAGd1[1,3] + PAs*DIAGd1[1,4] + P2s*DIAGd1[1,5] + PA2s*DIAGd1[1,6] + PEWs*EWPd1[1,0] + PEWcs*EWPd1[1,1]
    AB0sK0K0b= Ts*DIAGd1[2,0] + Cs*DIAGd1[2,1] + Ps*DIAGd1[2,2] + As*DIAGd1[2,3] + PAs*DIAGd1[2,4] + P2s*DIAGd1[2,5] + PA2s*DIAGd1[2,6] + PEWs*EWPd1[2,0] + PEWcs*EWPd1[2,1]
    AB0sPmPp = Ts*DIAGd1[3,0] + Cs*DIAGd1[3,1] + Ps*DIAGd1[3,2] + As*DIAGd1[3,3] + PAs*DIAGd1[3,4] + P2s*DIAGd1[3,5] + PA2s*DIAGd1[3,6] + PEWs*EWPd1[3,0] + PEWcs*EWPd1[3,1]
    AB0sKmKp = Ts*DIAGd1[4,0] + Cs*DIAGd1[4,1] + Ps*DIAGd1[4,2] + As*DIAGd1[4,3] + PAs*DIAGd1[4,4] + P2s*DIAGd1[4,5] + PA2s*DIAGd1[4,6] + PEWs*EWPd1[4,0] + PEWcs*EWPd1[4,1]
    AB0sP0P0 = Ts*DIAGd1[5,0] + Cs*DIAGd1[5,1] + Ps*DIAGd1[5,2] + As*DIAGd1[5,3] + PAs*DIAGd1[5,4] + P2s*DIAGd1[5,5] + PA2s*DIAGd1[5,6] + PEWs*EWPd1[5,0] + PEWcs*EWPd1[5,1]
    ABpPpK0  = Ts*DIAGd1[6,0] + Cs*DIAGd1[6,1] + Ps*DIAGd1[6,2] + As*DIAGd1[6,3] + PAs*DIAGd1[6,4] + P2s*DIAGd1[6,5] + PA2s*DIAGd1[6,6] + PEWs*EWPd1[6,0] + PEWcs*EWPd1[6,1]
    ABpP0Kp  = Ts*DIAGd1[7,0] + Cs*DIAGd1[7,1] + Ps*DIAGd1[7,2] + As*DIAGd1[7,3] + PAs*DIAGd1[7,4] + P2s*DIAGd1[7,5] + PA2s*DIAGd1[7,6] + PEWs*EWPd1[7,0] + PEWcs*EWPd1[7,1]

    #===Conjugate amplitudes===#
    AB0K0K0bb = Tdb*DIAGd0[0,0] + Cdb*DIAGd0[0,1] + Pdb*DIAGd0[0,2] + Adb*DIAGd0[0,3] + PAdb*DIAGd0[0,4] + P2db*DIAGd0[0,5] + PA2db*DIAGd0[0,6] + PEWdb*EWPd0[0,0] + PEWcdb*EWPd0[0,1]
    AB0PmPpb  = Tdb*DIAGd0[1,0] + Cdb*DIAGd0[1,1] + Pdb*DIAGd0[1,2] + Adb*DIAGd0[1,3] + PAdb*DIAGd0[1,4] + P2db*DIAGd0[1,5] + PA2db*DIAGd0[1,6] + PEWdb*EWPd0[1,0] + PEWcdb*EWPd0[1,1]
    AB0KmKpb  = Tdb*DIAGd0[2,0] + Cdb*DIAGd0[2,1] + Pdb*DIAGd0[2,2] + Adb*DIAGd0[2,3] + PAdb*DIAGd0[2,4] + P2db*DIAGd0[2,5] + PA2db*DIAGd0[2,6] + PEWdb*EWPd0[2,0] + PEWcdb*EWPd0[2,1]
    AB0P0P0b  = Tdb*DIAGd0[3,0] + Cdb*DIAGd0[3,1] + Pdb*DIAGd0[3,2] + Adb*DIAGd0[3,3] + PAdb*DIAGd0[3,4] + P2db*DIAGd0[3,5] + PA2db*DIAGd0[3,6] + PEWdb*EWPd0[3,0] + PEWcdb*EWPd0[3,1]
    AB0sPpKmb = Tdb*DIAGd0[4,0] + Cdb*DIAGd0[4,1] + Pdb*DIAGd0[4,2] + Adb*DIAGd0[4,3] + PAdb*DIAGd0[4,4] + P2db*DIAGd0[4,5] + PA2db*DIAGd0[4,6] + PEWdb*EWPd0[4,0] + PEWcdb*EWPd0[4,1]
    AB0sP0K0bb= Tdb*DIAGd0[5,0] + Cdb*DIAGd0[5,1] + Pdb*DIAGd0[5,2] + Adb*DIAGd0[5,3] + PAdb*DIAGd0[5,4] + P2db*DIAGd0[5,5] + PA2db*DIAGd0[5,6] + PEWdb*EWPd0[5,0] + PEWcdb*EWPd0[5,1]
    ABpK0bKpb = Tdb*DIAGd0[6,0] + Cdb*DIAGd0[6,1] + Pdb*DIAGd0[6,2] + Adb*DIAGd0[6,3] + PAdb*DIAGd0[6,4] + P2db*DIAGd0[6,5] + PA2db*DIAGd0[6,6] + PEWdb*EWPd0[6,0] + PEWcdb*EWPd0[6,1]
    ABpP0Ppb  = Tdb*DIAGd0[7,0] + Cdb*DIAGd0[7,1] + Pdb*DIAGd0[7,2] + Adb*DIAGd0[7,3] + PAdb*DIAGd0[7,4] + P2db*DIAGd0[7,5] + PA2db*DIAGd0[7,6] + PEWdb*EWPd0[7,0] + PEWcdb*EWPd0[7,1]
    
    AB0PmKpb  = Tsb*DIAGd1[0,0] + Csb*DIAGd1[0,1] + Psb*DIAGd1[0,2] + Asb*DIAGd1[0,3] + PAsb*DIAGd1[0,4] + P2sb*DIAGd1[0,5] + PA2sb*DIAGd1[0,6] + PEWsb*EWPd1[0,0] + PEWcsb*EWPd1[0,1]
    AB0P0K0b  = Tsb*DIAGd1[1,0] + Csb*DIAGd1[1,1] + Psb*DIAGd1[1,2] + Asb*DIAGd1[1,3] + PAsb*DIAGd1[1,4] + P2sb*DIAGd1[1,5] + PA2sb*DIAGd1[1,6] + PEWsb*EWPd1[1,0] + PEWcsb*EWPd1[1,1]
    AB0sK0K0bb= Tsb*DIAGd1[2,0] + Csb*DIAGd1[2,1] + Psb*DIAGd1[2,2] + Asb*DIAGd1[2,3] + PAsb*DIAGd1[2,4] + P2sb*DIAGd1[2,5] + PA2sb*DIAGd1[2,6] + PEWsb*EWPd1[2,0] + PEWcsb*EWPd1[2,1]
    AB0sPmPpb = Tsb*DIAGd1[3,0] + Csb*DIAGd1[3,1] + Psb*DIAGd1[3,2] + Asb*DIAGd1[3,3] + PAsb*DIAGd1[3,4] + P2sb*DIAGd1[3,5] + PA2sb*DIAGd1[3,6] + PEWsb*EWPd1[3,0] + PEWcsb*EWPd1[3,1]
    AB0sKmKpb = Tsb*DIAGd1[4,0] + Csb*DIAGd1[4,1] + Psb*DIAGd1[4,2] + Asb*DIAGd1[4,3] + PAsb*DIAGd1[4,4] + P2sb*DIAGd1[4,5] + PA2sb*DIAGd1[4,6] + PEWsb*EWPd1[4,0] + PEWcsb*EWPd1[4,1]
    AB0sP0P0b = Tsb*DIAGd1[5,0] + Csb*DIAGd1[5,1] + Psb*DIAGd1[5,2] + Asb*DIAGd1[5,3] + PAsb*DIAGd1[5,4] + P2sb*DIAGd1[5,5] + PA2sb*DIAGd1[5,6] + PEWsb*EWPd1[5,0] + PEWcsb*EWPd1[5,1]
    ABpPpK0b  = Tsb*DIAGd1[6,0] + Csb*DIAGd1[6,1] + Psb*DIAGd1[6,2] + Asb*DIAGd1[6,3] + PAsb*DIAGd1[6,4] + P2sb*DIAGd1[6,5] + PA2sb*DIAGd1[6,6] + PEWsb*EWPd1[6,0] + PEWcsb*EWPd1[6,1]
    ABpP0Kpb  = Tsb*DIAGd1[7,0] + Csb*DIAGd1[7,1] + Psb*DIAGd1[7,2] + Asb*DIAGd1[7,3] + PAsb*DIAGd1[7,4] + P2sb*DIAGd1[7,5] + PA2sb*DIAGd1[7,6] + PEWsb*EWPd1[7,0] + PEWcsb*EWPd1[7,1]


    #============= Branching Ratios =============#
    BrB0K0K0b   = (fB0K0K0b/(16*math.pi*mB0**2)) *(abs(AB0K0K0b)**2 + abs(AB0K0K0bb)**2)/GammaB0
    BrB0PmPp    = (fB0PpPm /(16*math.pi*mB0**2)) *(abs(AB0PmPp)**2  + abs(AB0PmPpb)**2)/GammaB0
    BrB0KmKp    = (fB0KpKm /(16*math.pi*mB0**2)) *(abs(AB0KmKp)**2  + abs(AB0KmKpb)**2)/GammaB0
    BrB0P0P0    = (fB0P0P0 /(16*math.pi*mB0**2)) *(abs(AB0P0P0)**2  + abs(AB0P0P0b)**2)/GammaB0
    BrB0sPpKm   = (fB0sPpKm/(16*math.pi*mB0s**2))*(abs(AB0sPpKm)**2 + abs(AB0sPpKmb)**2)/GammaB0s
    #BrB0sP0K0b = (fB0sP0K0b/(16*math.pi*mB0s**2))*(abs(AB0sP0K0b)**2+ abs(AB0sP0K0bb)**2)
    BrBpK0bKp   = (fBpK0bKp/(16*math.pi*mBp**2)) *(abs(ABpK0bKp)**2 + abs(ABpK0bKpb)**2)/GammaBp
    BrBpP0Pp    = (fBpP0Pp /(16*math.pi*mBp**2)) *(abs(ABpP0Pp)**2  + abs(ABpP0Ppb)**2)/GammaBp

    BrB0PmKp    = (fB0PmKp /(16*math.pi*mB0**2)) *(abs(AB0PmKp)**2  + abs(AB0PmKpb)**2)/GammaB0
    BrB0P0K0    = (fB0P0K0 /(16*math.pi*mB0**2)) *(abs(AB0P0K0)**2  + abs(AB0P0K0b)**2)/GammaB0
    BrB0sK0K0b  = (fB0sK0K0b/(16*math.pi*mB0s**2))*(abs(AB0sK0K0b)**2+abs(AB0sK0K0bb)**2)/GammaB0s
    BrB0sPmPp   = (fB0sPpPm/(16*math.pi*mB0s**2)) *(abs(AB0sPmPp)**2+ abs(AB0sPmPpb)**2)/GammaB0s
    BrB0sKmKp   = (fB0sKpKm/(16*math.pi*mB0s**2)) *(abs(AB0sKmKp)**2+ abs(AB0sKmKpb)**2)/GammaB0s
    BrB0sP0P0   = (fB0sP0P0/(16*math.pi*mB0s**2)) *(abs(AB0sP0P0)**2+ abs(AB0sP0P0b)**2)/GammaB0s
    BrBpPpK0    = (fBpPpK0 /(16*math.pi*mBp**2))  *(abs(ABpPpK0)**2 + abs(ABpPpK0b)**2)/GammaBp
    BrBpP0Kp    = (fBpP0Kp /(16*math.pi*mBp**2))  *(abs(ABpP0Kp)**2 + abs(ABpP0Kpb)**2)/GammaBp

    #============= Direct CP asymetries ============#
    aCPB0K0K0b  = (abs(AB0K0K0bb)**2 - abs(AB0K0K0b)**2) / (abs(AB0K0K0bb)**2 + abs(AB0K0K0b)**2)
    aCPB0PmPp   = (abs(AB0PmPpb)**2  - abs(AB0PmPp)**2)  / (abs(AB0PmPpb)**2  + abs(AB0PmPp)**2)
    aCPB0P0P0   = (abs(AB0P0P0b)**2  - abs(AB0P0P0)**2)  / (abs(AB0P0P0b)**2  + abs(AB0P0P0)**2)
    aCPB0sPpKm  = (abs(AB0sPpKmb)**2 - abs(AB0sPpKm)**2) / (abs(AB0sPpKmb)**2 + abs(AB0sPpKm)**2)
    aCPBpK0bKp  = (abs(ABpK0bKpb)**2 - abs(ABpK0bKp)**2) / (abs(ABpK0bKpb)**2 + abs(ABpK0bKp)**2)
    aCPBpP0Pp   = (abs(ABpP0Ppb)**2  - abs(ABpP0Pp)**2)  / (abs(ABpP0Ppb)**2  + abs(ABpP0Pp)**2)

    aCPB0PmKp   = (abs(AB0PmKpb)**2  - abs(AB0PmKp)**2)  / (abs(AB0PmKpb)**2  + abs(AB0PmKp)**2)
    aCPB0P0K0   = (abs(AB0P0K0b)**2  - abs(AB0P0K0)**2)  / (abs(AB0P0K0b)**2  + abs(AB0P0K0)**2)
    aCPB0sKmKp  = (abs(AB0sKmKpb)**2 - abs(AB0sKmKp)**2) / (abs(AB0sKmKpb)**2 + abs(AB0sKmKp)**2)
    aCPBpPpK0   = (abs(ABpPpK0b)**2  - abs(ABpPpK0)**2)  / (abs(ABpPpK0b)**2  + abs(ABpPpK0)**2)
    aCPBpP0Kp   = (abs(ABpP0Kpb)**2  - abs(ABpP0Kp)**2)  / (abs(ABpP0Kpb)**2  + abs(ABpP0Kp)**2)

    #========== Indirect CP asymetries ============#
    sCPB0PmPp   = 2*(np.exp(-2j*beta)*np.conjugate(AB0PmPp)*AB0PmPpb)/(abs(AB0PmPpb)**2 + abs(AB0PmPp)**2)
    sCPB0P0K0   = 2*(np.exp(-2j*beta)*np.conjugate(AB0P0K0)*AB0P0K0b)/(abs(AB0P0K0b)**2 + abs(AB0P0K0)**2)
    sCPB0K0K0b  = 2*(np.exp(-2j*beta)*np.conjugate(AB0K0K0b)*AB0K0K0bb)/(abs(AB0K0K0b)**2 + abs(AB0K0K0bb)**2)
    sCPB0sKmKp  = 2*(np.exp(-2j*betas)*np.conjugate(AB0sKmKp)*AB0sKmKpb)/(abs(AB0sKmKpb)**2 + abs(AB0sKmKp)**2)

    ImsCPB0PmPp = sCPB0PmPp.imag
    ImsCPB0P0K0 = sCPB0P0K0.imag
    ImsCPB0K0K0b= sCPB0K0K0b.imag
    ImsCPB0sKmKp= sCPB0sKmKp.imag

    #========== Chi squared ==========#
    chi2_BR = (BrB0K0K0b - B0K0K0b_exp)**2 / (B0K0K0b_inc)**2 + (BrB0PmPp - B0PpPm_exp)**2 / (B0PpPm_inc)**2  +\
              (BrB0KmKp  - B0KpKm_exp)**2  / (B0KpKm_inc)**2  + (BrB0P0P0 - B0P0P0_exp)**2 / (B0P0P0_inc)**2  +\
              (BrB0sPpKm - B0sPpKm_exp)**2 / (B0sPpKm_inc)**2 + (BrBpK0bKp- BpK0bKp_exp)**2/ (BpK0bKp_inc)**2 +\
              (BrBpP0Pp  - BpP0Pp_exp)**2  / (BpP0Pp_inc)**2  + (BrB0PmKp - B0PmKp_exp)**2 / (B0PmKp_inc)**2  +\
              (BrB0P0K0  - B0P0K0_exp)**2  / (B0P0K0_inc)**2  + (BrB0sK0K0b-B0sK0K0b_exp)**2/(B0sK0K0b_inc)**2+\
              (BrB0sPmPp - B0sPpPm_exp)**2 / (B0sPpPm_inc)**2 + (BrB0sKmKp- B0sKpKm_exp)**2/ (B0sKpKm_inc)**2 +\
              (BrB0sP0P0 - B0sP0P0_exp)**2 / (B0sP0P0_inc)**2 + (BrBpPpK0 - BpPpK0_exp)**2 / (BpPpK0_inc)**2  +\
              (BrBpP0Kp  - BpP0Kp_exp)**2  / (BpP0Kp_inc)**2
    
    chi2ACP = (aCPB0K0K0b- ACP_B0K0K0b_exp)**2 / (ACP_B0K0K0b_inc)**2   + (aCPB0PmPp - ACP_B0PpPm_exp)**2 / (ACP_B0PpPm_inc)**2  +\
              (aCPB0P0P0 - ACP_B0P0P0_exp)**2  / (ACP_B0P0P0_inc)**2    + (aCPB0sPpKm- ACP_B0sPpKm_exp)**2/ (ACP_B0sPpKm_inc)**2 +\
              (aCPBpK0bKp- ACP_BpK0bKp_exp)**2 / (ACP_BpK0bKp_inc)**2   + (aCPBpP0Pp - ACP_BpP0Pp_exp)**2 / (ACP_BpP0Pp_inc)**2  +\
              (aCPB0PmKp - ACP_B0PmKp_exp)**2  / (ACP_B0PmKp_inc)**2    + (aCPB0P0K0  - ACP_B0P0K0_exp)**2 / (ACP_B0P0K0_inc)**2  +\
              (aCPB0sKmKp- ACP_B0sKpKm_exp)**2 / (ACP_B0sKpKm_inc)**2   + (aCPBpPpK0 - ACP_BpPpK0_exp)**2 / (ACP_BpPpK0_inc)**2  +\
              (aCPBpP0Kp - ACP_BpP0Kp_exp)**2  / (ACP_BpP0Kp_inc)**2
    
    chi2SCP = (ImsCPB0PmPp-SCP_B0PpPm_exp)**2/(SCP_B0PpPm_inc)**2+(ImsCPB0sKmKp-SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc)**2 +(ImsCPB0P0K0-SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc)**2 + (ImsCPB0K0K0b-SCP_B0K0K0b_exp)**2/(SCP_B0K0K0b_inc)**2

    chi2CKM = (Vud - Vud_exp)**2/(Vud_inc)**2 + (Vus - Vus_exp)**2/(Vus_inc)**2 + (Vub - Vub_exp)**2/(Vub_inc)**2 +\
              (Vtd - Vtd_exp)**2/(Vtd_inc)**2 + (Vts - Vts_exp)**2/(Vts_inc)**2 + (Vtb - Vtb_exp)**2/(Vtb_inc)**2
    
    chi2Gam = (gamma - gamma_exp)**2/(gamma_inc)**2
    chi2Bet = (beta - beta_exp)**2/(beta_inc)**2 + (betas - betas_exp)**2/(betas_inc)**2

    chi2total = chi2_BR + chi2ACP + chi2SCP + chi2CKM + chi2Gam + chi2Bet
    print(ImsCPB0PmPp,ImsCPB0K0K0b,ImsCPB0P0K0,ImsCPB0sKmKp)
    return chi2total

def chi2Printed(parameters):
    ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1 = parameters
    
    
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    print(amplitudes_DeltaS08X8[3])
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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    print(Im_SCP_B0PpPm,Im_SCP_B0K0K0b,Im_SCP_B0P0K0,Im_SCP_B0sKpKm)

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return -chi2_total

def chi2WithoutEta(parameters):
    ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1 = parameters
    
    
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm 

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return -chi2_total

def chi2Minuit(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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
    return chi2_total

def chi2MinuitWithoutEta(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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


    #===EWP Diagrams===#
    PEWTd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWCd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWAd8X8 = -3/2.0 * (c9+c10)/(c1+c2)*AA8X8*Vtbd



    #===Delta S = 1===#
    TTs8X8 = TT8X8*Vubs
    CCs8X8 = CC8X8*Vubs
    PPucs8X8 = PPuc8X8*Vubs
    AAs8X8 = AA8X8*Vubs
    PAucs8X8 = PAuc8X8*Vubs
    PPtcs8X8 = PPtc8X8*Vtbs
    PAtcs8X8 = PAtc8X8*Vtbs



    #===EWP Diagrams===#
    PEWTs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWCs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWAs8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbs


    #===Diagrams with conjugate weak phases===#
    TTdb8X8 = TT8X8*Vubdb
    CCdb8X8 = CC8X8*Vubdb
    PPucdb8X8 = PPuc8X8*Vubdb
    AAdb8X8 = AA8X8*Vubdb
    PAucdb8X8 = PAuc8X8*Vubdb
    PPtcdb8X8 = PPtc8X8*Vtbdb
    PAtcdb8X8 = PAtc8X8*Vtbdb


    #===EWP Diagrams===#
    PEWTdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWCdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWAdb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbdb


    #===Delta S = 1===#
    TTsb8X8 = TT8X8*Vubsb
    CCsb8X8 = CC8X8*Vubsb
    PPucsb8X8 = PPuc8X8*Vubsb
    AAsb8X8 = AA8X8*Vubsb
    PAucsb8X8 = PAuc8X8*Vubsb
    PPtcsb8X8 = PPtc8X8*Vtbsb
    PAtcsb8X8 = PAtc8X8*Vtbsb

    #===EWP Diagrams===#
    PEWTsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWCsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWAsb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbsb

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


    #===Vector of diagrams===#
    diagramsd08X8 = np.array([TTd8X8,CCd8X8,PPucd8X8,AAd8X8,PAucd8X8,PPtcd8X8,PAtcd8X8,PEWTd8X8,PEWCd8X8,PEWAd8X8])
    diagramsd18X8 = np.array([TTs8X8,CCs8X8,PPucs8X8,AAs8X8,PAucs8X8,PPtcs8X8,PAtcs8X8,PEWTs8X8,PEWCs8X8,PEWAs8X8])
    diagramsd0b8X8= np.array([TTdb8X8,CCdb8X8,PPucdb8X8,AAdb8X8,PAucdb8X8,PPtcdb8X8,PAtcdb8X8,PEWTdb8X8,PEWCdb8X8,PEWAdb8X8])
    diagramsd1b8X8= np.array([TTsb8X8,CCsb8X8,PPucsb8X8,AAsb8X8,PAucsb8X8,PPtcsb8X8,PAtcsb8X8,PEWTsb8X8,PEWCsb8X8,PEWAsb8X8])

    #===Amplitudes===#
    #===Initialising amplitudes===#
    amplitudes_DeltaS08X8  = np.zeros(12)
    amplitudes_DeltaS18X8  = np.zeros(12)
    amplitudes_DeltaS0b8X8 = np.zeros(12)
    amplitudes_DeltaS1b8X8 = np.zeros(12)

    #===Calculating amplitudes===#
    amplitudes_DeltaS08X8 = np.matmul(DIAGd08X8,diagramsd08X8)
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS0b8X8= np.matmul(DIAGd08X8,diagramsd0b8X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    #===Calculating branching ratios===#
    #===Branching ratios Delta S = 0===#
    BR_BpK0bKp = (fBpK0bKp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[0])**2 + abs(amplitudes_DeltaS0b8X8[0])**2)/GammaBp
    BR_BpP0Pp  = (fBpP0Pp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[1])**2 + abs(amplitudes_DeltaS0b8X8[1])**2)/GammaBp
    BR_B0K0K0b = (fB0K0K0b/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[3])**2 +abs(amplitudes_DeltaS0b8X8[3])**2)/GammaB0
    BR_B0PpPm  = (fB0PpPm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[4])**2 + abs(amplitudes_DeltaS0b8X8[4])**2)/GammaB0
    BR_B0KpKm  = (fB0KpKm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[5])**2 + abs(amplitudes_DeltaS0b8X8[5])**2)/GammaB0
    BR_B0P0P0  = (fB0P0P0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[6])**2 + abs(amplitudes_DeltaS0b8X8[6])**2)/GammaB0
    BR_B0sPpKm = (fB0sPpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[9])**2 + abs(amplitudes_DeltaS0b8X8[9])**2)/GammaB0s
    BR_B0sP0K0b= (fB0sP0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[10])**2 + abs(amplitudes_DeltaS0b8X8[10])**2)/GammaB0s
  
    #===Branching ratios Delta S = 1===#
    BR_BpPpK0  = (fBpPpK0/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[0])**2 + abs(amplitudes_DeltaS1b8X8[0])**2)/GammaBp
    BR_BpP0Kp  = (fBpP0Kp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[1])**2 + abs(amplitudes_DeltaS1b8X8[1])**2)/GammaBp
    BR_B0PmKp  = (fB0PmKp/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[3])**2 + abs(amplitudes_DeltaS1b8X8[3])**2)/GammaB0
    BR_B0P0K0  = (fB0P0K0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[4])**2 + abs(amplitudes_DeltaS1b8X8[4])**2)/GammaB0
    BR_B0sK0K0b= (fB0sK0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[6])**2 + abs(amplitudes_DeltaS1b8X8[6])**2)/GammaB0s
    BR_B0sPpPm = (fB0sPpPm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[7])**2 + abs(amplitudes_DeltaS1b8X8[7])**2)/GammaB0s
    BR_B0sKpKm = (fB0sKpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[8])**2 + abs(amplitudes_DeltaS1b8X8[8])**2)/GammaB0s
    BR_B0sP0P0 = (fB0sP0P0/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[9])**2 + abs(amplitudes_DeltaS1b8X8[9])**2)/GammaB0s

    #===Direct CP asymmetries Delta S = 0===#
    ACP_BpK0bKp = (abs(amplitudes_DeltaS0b8X8[0])**2 - abs(amplitudes_DeltaS08X8[0])**2)/(abs(amplitudes_DeltaS0b8X8[0])**2 + abs(amplitudes_DeltaS08X8[0])**2)
    ACP_BpP0Pp  = (abs(amplitudes_DeltaS0b8X8[1])**2 - abs(amplitudes_DeltaS08X8[1])**2)/(abs(amplitudes_DeltaS0b8X8[1])**2 + abs(amplitudes_DeltaS08X8[1])**2)
    ACP_B0K0K0b = (abs(amplitudes_DeltaS0b8X8[3])**2 - abs(amplitudes_DeltaS08X8[3]**2))/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    ACP_B0PpPm  = (abs(amplitudes_DeltaS0b8X8[4])**2 - abs(amplitudes_DeltaS08X8[4])**2)/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    ACP_B0KpKm  = (abs(amplitudes_DeltaS0b8X8[5])**2 - abs(amplitudes_DeltaS08X8[5])**2)/(abs(amplitudes_DeltaS0b8X8[5])**2 + abs(amplitudes_DeltaS08X8[5])**2)
    ACP_B0P0P0  = (abs(amplitudes_DeltaS0b8X8[6])**2 - abs(amplitudes_DeltaS08X8[6])**2)/(abs(amplitudes_DeltaS0b8X8[6])**2 + abs(amplitudes_DeltaS08X8[6])**2)
    ACP_B0sPpKm = (abs(amplitudes_DeltaS0b8X8[9])**2 - abs(amplitudes_DeltaS08X8[9])**2)/(abs(amplitudes_DeltaS0b8X8[9])**2 + abs(amplitudes_DeltaS08X8[9])**2)
    ACP_B0sP0K0b= (abs(amplitudes_DeltaS0b8X8[10])**2 - abs(amplitudes_DeltaS08X8[10])**2)/(abs(amplitudes_DeltaS0b8X8[10])**2 + abs(amplitudes_DeltaS08X8[10])**2)
    
    #===Direct CP asymmetries Delta S = 1===#
    ACP_BpPpK0  = (abs(amplitudes_DeltaS1b8X8[0])**2 - abs(amplitudes_DeltaS18X8[0])**2)/(abs(amplitudes_DeltaS1b8X8[0])**2 + abs(amplitudes_DeltaS18X8[0])**2)
    ACP_BpP0Kp  = (abs(amplitudes_DeltaS1b8X8[1])**2 - abs(amplitudes_DeltaS18X8[1])**2)/(abs(amplitudes_DeltaS1b8X8[1])**2 + abs(amplitudes_DeltaS18X8[1])**2)
    ACP_B0PmKp  = (abs(amplitudes_DeltaS1b8X8[3])**2 - abs(amplitudes_DeltaS18X8[3])**2)/(abs(amplitudes_DeltaS1b8X8[3])**2 + abs(amplitudes_DeltaS18X8[3])**2)
    ACP_B0P0K0  = (abs(amplitudes_DeltaS1b8X8[4])**2 - abs(amplitudes_DeltaS18X8[4])**2)/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    ACP_B0sK0K0b= (abs(amplitudes_DeltaS1b8X8[6])**2 - abs(amplitudes_DeltaS18X8[6])**2)/(abs(amplitudes_DeltaS1b8X8[6])**2 + abs(amplitudes_DeltaS18X8[6])**2)
    ACP_B0sPpPm = (abs(amplitudes_DeltaS1b8X8[7])**2 - abs(amplitudes_DeltaS18X8[7])**2)/(abs(amplitudes_DeltaS1b8X8[7])**2 + abs(amplitudes_DeltaS18X8[7])**2)
    ACP_B0sKpKm = (abs(amplitudes_DeltaS1b8X8[8])**2 - abs(amplitudes_DeltaS18X8[8])**2)/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    ACP_B0sP0P0 = (abs(amplitudes_DeltaS1b8X8[9])**2 - abs(amplitudes_DeltaS18X8[9])**2)/(abs(amplitudes_DeltaS1b8X8[9])**2 + abs(amplitudes_DeltaS18X8[9])**2)

    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag

    #===Chi2 Contributions===#
    #===Branching ratios===#
    chi2_BR_BpK0bKp = (BR_BpK0bKp - BpK0bKp_exp)**2/(BpK0bKp_inc**2)
    chi2_BR_BpP0Pp  = (BR_BpP0Pp - BpP0Pp_exp)**2/(BpP0Pp_inc**2)
    chi2_BR_B0K0K0b = (BR_B0K0K0b - B0K0K0b_exp)**2/(B0K0K0b_inc**2)
    chi2_BR_B0PpPm  = (BR_B0PpPm - B0PpPm_exp)**2/(B0PpPm_inc**2)
    chi2_BR_B0KpKm  = (BR_B0KpKm - B0KpKm_exp)**2/(B0KpKm_inc**2)
    chi2_BR_B0P0P0  = (BR_B0P0P0 - B0P0P0_exp)**2/(B0P0P0_inc**2)
    chi2_BR_B0sPpKm = (BR_B0sPpKm - B0sPpKm_exp)**2/(B0sPpKm_inc**2)
    #chi2_BR_B0sP0K0b= (BR_B0sP0K0b - B0sP0K0b_exp)**2/(B0sP0K0b_inc**2)

    chi2_BR_BpPpK0  = (BR_BpPpK0 - BpPpK0_exp)**2/(BpPpK0_inc**2)
    chi2_BR_BpP0Kp  = (BR_BpP0Kp - BpP0Kp_exp)**2/(BpP0Kp_inc**2)
    chi2_BR_B0PmKp  = (BR_B0PmKp - B0PmKp_exp)**2/(B0PmKp_inc**2)
    chi2_BR_B0P0K0  = (BR_B0P0K0 - B0P0K0_exp)**2/(B0P0K0_inc**2)
    chi2_BR_B0sK0K0b= (BR_B0sK0K0b - B0sK0K0b_exp)**2/(B0sK0K0b_inc**2)
    chi2_BR_B0sPpPm = (BR_B0sPpPm - B0sPpPm_exp)**2/(B0sPpPm_inc**2)
    chi2_BR_B0sKpKm = (BR_B0sKpKm - B0sKpKm_exp)**2/(B0sKpKm_inc**2)
    chi2_BR_B0sP0P0 = (BR_B0sP0P0 - B0sP0P0_exp)**2/(B0sP0P0_inc**2)

    #===Direct CP asymmetries===#
    chi2_ACP_BpK0bKp = (ACP_BpK0bKp - ACP_BpK0bKp_exp)**2/(ACP_BpK0bKp_inc**2)
    chi2_ACP_BpP0Pp  = (ACP_BpP0Pp - ACP_BpP0Pp_exp)**2/(ACP_BpP0Pp_inc**2)
    chi2_ACP_B0K0K0b = (ACP_B0K0K0b - ACP_B0K0K0b_exp)**2/(ACP_B0K0K0b_inc**2)
    chi2_ACP_B0PpPm  = (ACP_B0PpPm - ACP_B0PpPm_exp)**2/(ACP_B0PpPm_inc**2)
    #chi2_ACP_B0KpKm  = (ACP_B0KpKm - ACP_B0KpKm_exp)**2/(ACP_B0KpKm_inc**2)
    chi2_ACP_B0P0P0  = (ACP_B0P0P0 - ACP_B0P0P0_exp)**2/(ACP_B0P0P0_inc**2)
    chi2_ACP_B0sPpKm = (ACP_B0sPpKm - ACP_B0sPpKm_exp)**2/(ACP_B0sPpKm_inc**2)
    #chi2_ACP_B0sP0K0b= (ACP_B0sP0K0b - ACP_B0sP0K0b_exp)**2/(ACP_B0sP0K0b_inc**2)

    chi2_ACP_BpPpK0  = (ACP_BpPpK0 - ACP_BpPpK0_exp)**2/(ACP_BpPpK0_inc**2)
    chi2_ACP_BpP0Kp  = (ACP_BpP0Kp - ACP_BpP0Kp_exp)**2/(ACP_BpP0Kp_inc**2)
    chi2_ACP_B0PmKp  = (ACP_B0PmKp - ACP_B0PmKp_exp)**2/(ACP_B0PmKp_inc**2)
    chi2_ACP_B0P0K0  = (ACP_B0P0K0 - ACP_B0P0K0_exp)**2/(ACP_B0P0K0_inc**2)
    #chi2_ACP_B0sK0K0b= (ACP_B0sK0K0b - ACP_B0sK0K0b_exp)**2/(ACP_B0sK0K0b_inc**2)
    #chi2_ACP_B0sPpPm = (ACP_B0sPpPm - ACP_B0sPpPm_exp)**2/(ACP_B0sPpPm_inc**2)
    chi2_ACP_B0sKpKm = (ACP_B0sKpKm - ACP_B0sKpKm_exp)**2/(ACP_B0sKpKm_inc**2)
    #chi2_ACP_B0sP0P0 = (ACP_B0sP0P0 - ACP_B0sP0P0_exp)**2/(ACP_B0sP0P0_inc**2)

    #===Indirect CP asymmetries===#
    chi2_SCP_B0K0K0b    = (Im_SCP_B0K0K0b - SCP_B0K0K0b_exp)**2/(SCP_B0K0K0b_inc**2)
    chi2_SCP_B0PpPm     = (Im_SCP_B0PpPm - SCP_B0PpPm_exp)**2/(SCP_B0PpPm_inc**2)

    chi2_SCP_B0P0K0     = (Im_SCP_B0P0K0 - SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc**2)
    chi2_SCP_B0sKpKm    = (Im_SCP_B0sKpKm - SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc**2)

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm 

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total

def chi2MinuitWithoutEtaDS0only(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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


    #===EWP Diagrams===#
    PEWTd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWCd8X8 = -3/4.0 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbd
    PEWAd8X8 = -3/2.0 * (c9+c10)/(c1+c2)*AA8X8*Vtbd



    #===Delta S = 1===#
    TTs8X8 = TT8X8*Vubs
    CCs8X8 = CC8X8*Vubs
    PPucs8X8 = PPuc8X8*Vubs
    AAs8X8 = AA8X8*Vubs
    PAucs8X8 = PAuc8X8*Vubs
    PPtcs8X8 = PPtc8X8*Vtbs
    PAtcs8X8 = PAtc8X8*Vtbs



    #===EWP Diagrams===#
    PEWTs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWCs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWAs8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbs


    #===Diagrams with conjugate weak phases===#
    TTdb8X8 = TT8X8*Vubdb
    CCdb8X8 = CC8X8*Vubdb
    PPucdb8X8 = PPuc8X8*Vubdb
    AAdb8X8 = AA8X8*Vubdb
    PAucdb8X8 = PAuc8X8*Vubdb
    PPtcdb8X8 = PPtc8X8*Vtbdb
    PAtcdb8X8 = PAtc8X8*Vtbdb


    #===EWP Diagrams===#
    PEWTdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWCdb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbdb
    PEWAdb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbdb


    #===Delta S = 1===#
    TTsb8X8 = TT8X8*Vubsb
    CCsb8X8 = CC8X8*Vubsb
    PPucsb8X8 = PPuc8X8*Vubsb
    AAsb8X8 = AA8X8*Vubsb
    PAucsb8X8 = PAuc8X8*Vubsb
    PPtcsb8X8 = PPtc8X8*Vtbsb
    PAtcsb8X8 = PAtc8X8*Vtbsb

    #===EWP Diagrams===#
    PEWTsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWCsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWAsb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbsb

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


    #===Vector of diagrams===#
    diagramsd08X8 = np.array([TTd8X8,CCd8X8,PPucd8X8,AAd8X8,PAucd8X8,PPtcd8X8,PAtcd8X8,PEWTd8X8,PEWCd8X8,PEWAd8X8])
    diagramsd0b8X8= np.array([TTdb8X8,CCdb8X8,PPucdb8X8,AAdb8X8,PAucdb8X8,PPtcdb8X8,PAtcdb8X8,PEWTdb8X8,PEWCdb8X8,PEWAdb8X8])

    #===Amplitudes===#
    #===Initialising amplitudes===#
    amplitudes_DeltaS08X8  = np.zeros(12)
    amplitudes_DeltaS0b8X8 = np.zeros(12)

    #===Calculating amplitudes===#
    amplitudes_DeltaS08X8 = np.matmul(DIAGd08X8,diagramsd08X8)
    amplitudes_DeltaS0b8X8= np.matmul(DIAGd08X8,diagramsd0b8X8)

    #===Calculating branching ratios===#
    #===Branching ratios Delta S = 0===#
    BR_BpK0bKp = (fBpK0bKp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[0])**2 + abs(amplitudes_DeltaS0b8X8[0])**2)/GammaBp
    BR_BpP0Pp  = (fBpP0Pp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS08X8[1])**2 + abs(amplitudes_DeltaS0b8X8[1])**2)/GammaBp
    BR_B0K0K0b = (fB0K0K0b/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[3])**2 +abs(amplitudes_DeltaS0b8X8[3])**2)/GammaB0
    BR_B0PpPm  = (fB0PpPm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[4])**2 + abs(amplitudes_DeltaS0b8X8[4])**2)/GammaB0
    BR_B0KpKm  = (fB0KpKm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[5])**2 + abs(amplitudes_DeltaS0b8X8[5])**2)/GammaB0
    BR_B0P0P0  = (fB0P0P0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[6])**2 + abs(amplitudes_DeltaS0b8X8[6])**2)/GammaB0
    BR_B0sPpKm = (fB0sPpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[9])**2 + abs(amplitudes_DeltaS0b8X8[9])**2)/GammaB0s
    BR_B0sP0K0b= (fB0sP0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[10])**2 + abs(amplitudes_DeltaS0b8X8[10])**2)/GammaB0s


    #===Direct CP asymmetries Delta S = 0===#
    ACP_BpK0bKp = (abs(amplitudes_DeltaS0b8X8[0])**2 - abs(amplitudes_DeltaS08X8[0])**2)/(abs(amplitudes_DeltaS0b8X8[0])**2 + abs(amplitudes_DeltaS08X8[0])**2)
    ACP_BpP0Pp  = (abs(amplitudes_DeltaS0b8X8[1])**2 - abs(amplitudes_DeltaS08X8[1])**2)/(abs(amplitudes_DeltaS0b8X8[1])**2 + abs(amplitudes_DeltaS08X8[1])**2)
    ACP_B0K0K0b = (abs(amplitudes_DeltaS0b8X8[3])**2 - abs(amplitudes_DeltaS08X8[3]**2))/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    ACP_B0PpPm  = (abs(amplitudes_DeltaS0b8X8[4])**2 - abs(amplitudes_DeltaS08X8[4])**2)/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    ACP_B0KpKm  = (abs(amplitudes_DeltaS0b8X8[5])**2 - abs(amplitudes_DeltaS08X8[5])**2)/(abs(amplitudes_DeltaS0b8X8[5])**2 + abs(amplitudes_DeltaS08X8[5])**2)
    ACP_B0P0P0  = (abs(amplitudes_DeltaS0b8X8[6])**2 - abs(amplitudes_DeltaS08X8[6])**2)/(abs(amplitudes_DeltaS0b8X8[6])**2 + abs(amplitudes_DeltaS08X8[6])**2)
    ACP_B0sPpKm = (abs(amplitudes_DeltaS0b8X8[9])**2 - abs(amplitudes_DeltaS08X8[9])**2)/(abs(amplitudes_DeltaS0b8X8[9])**2 + abs(amplitudes_DeltaS08X8[9])**2)
    ACP_B0sP0K0b= (abs(amplitudes_DeltaS0b8X8[10])**2 - abs(amplitudes_DeltaS08X8[10])**2)/(abs(amplitudes_DeltaS0b8X8[10])**2 + abs(amplitudes_DeltaS08X8[10])**2)
    
    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag


    #===Chi2 Contributions===#
    #===Branching ratios===#
    chi2_BR_BpK0bKp = (BR_BpK0bKp - BpK0bKp_exp)**2/(BpK0bKp_inc**2)
    chi2_BR_BpP0Pp  = (BR_BpP0Pp - BpP0Pp_exp)**2/(BpP0Pp_inc**2)
    chi2_BR_B0K0K0b = (BR_B0K0K0b - B0K0K0b_exp)**2/(B0K0K0b_inc**2)
    chi2_BR_B0PpPm  = (BR_B0PpPm - B0PpPm_exp)**2/(B0PpPm_inc**2)
    chi2_BR_B0KpKm  = (BR_B0KpKm - B0KpKm_exp)**2/(B0KpKm_inc**2)
    chi2_BR_B0P0P0  = (BR_B0P0P0 - B0P0P0_exp)**2/(B0P0P0_inc**2)
    chi2_BR_B0sPpKm = (BR_B0sPpKm - B0sPpKm_exp)**2/(B0sPpKm_inc**2)
    #chi2_BR_B0sP0K0b= (BR_B0sP0K0b - B0sP0K0b_exp)**2/(B0sP0K0b_inc**2)

    #===Direct CP asymmetries===#
    chi2_ACP_BpK0bKp = (ACP_BpK0bKp - ACP_BpK0bKp_exp)**2/(ACP_BpK0bKp_inc**2)
    chi2_ACP_BpP0Pp  = (ACP_BpP0Pp - ACP_BpP0Pp_exp)**2/(ACP_BpP0Pp_inc**2)
    chi2_ACP_B0K0K0b = (ACP_B0K0K0b - ACP_B0K0K0b_exp)**2/(ACP_B0K0K0b_inc**2)
    chi2_ACP_B0PpPm  = (ACP_B0PpPm - ACP_B0PpPm_exp)**2/(ACP_B0PpPm_inc**2)
    #chi2_ACP_B0KpKm  = (ACP_B0KpKm - ACP_B0KpKm_exp)**2/(ACP_B0KpKm_inc**2)
    chi2_ACP_B0P0P0  = (ACP_B0P0P0 - ACP_B0P0P0_exp)**2/(ACP_B0P0P0_inc**2)
    chi2_ACP_B0sPpKm = (ACP_B0sPpKm - ACP_B0sPpKm_exp)**2/(ACP_B0sPpKm_inc**2)
    #chi2_ACP_B0sP0K0b= (ACP_B0sP0K0b - ACP_B0sP0K0b_exp)**2/(ACP_B0sP0K0b_inc**2)


    #===Indirect CP asymmetries===#
    chi2_SCP_B0K0K0b    = (Im_SCP_B0K0K0b - SCP_B0K0K0b_exp)**2/(SCP_B0K0K0b_inc**2)
    chi2_SCP_B0PpPm     = (Im_SCP_B0PpPm - SCP_B0PpPm_exp)**2/(SCP_B0PpPm_inc**2)


    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0sPpKm
              
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total

def chi2MinuitWithoutEtaDS1only(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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

    
    #===Diagrams with weak phases===#
    #=== 8X8 Final states ===#
    #===Delta S = 1===#
    TTs8X8 = TT8X8*Vubs
    CCs8X8 = CC8X8*Vubs
    PPucs8X8 = PPuc8X8*Vubs
    AAs8X8 = AA8X8*Vubs
    PAucs8X8 = PAuc8X8*Vubs
    PPtcs8X8 = PPtc8X8*Vtbs
    PAtcs8X8 = PAtc8X8*Vtbs



    #===EWP Diagrams===#
    PEWTs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWCs8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbs
    PEWAs8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbs


    #===Delta S = 1===#
    TTsb8X8 = TT8X8*Vubsb
    CCsb8X8 = CC8X8*Vubsb
    PPucsb8X8 = PPuc8X8*Vubsb
    AAsb8X8 = AA8X8*Vubsb
    PAucsb8X8 = PAuc8X8*Vubsb
    PPtcsb8X8 = PPtc8X8*Vtbsb
    PAtcsb8X8 = PAtc8X8*Vtbsb

    #===EWP Diagrams===#
    PEWTsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWCsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWAsb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbsb

    #===Diagram contributions to amplitudes===#
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


    #===Vector of diagrams===#
    diagramsd18X8 = np.array([TTs8X8,CCs8X8,PPucs8X8,AAs8X8,PAucs8X8,PPtcs8X8,PAtcs8X8,PEWTs8X8,PEWCs8X8,PEWAs8X8])
    diagramsd1b8X8= np.array([TTsb8X8,CCsb8X8,PPucsb8X8,AAsb8X8,PAucsb8X8,PPtcsb8X8,PAtcsb8X8,PEWTsb8X8,PEWCsb8X8,PEWAsb8X8])

    #===Amplitudes===#
    #===Initialising amplitudes===#
    amplitudes_DeltaS18X8  = np.zeros(12)
    amplitudes_DeltaS1b8X8 = np.zeros(12)

    #===Calculating amplitudes===#
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    #===Calculating branching ratios===#  
    #===Branching ratios Delta S = 1===#
    BR_BpPpK0  = (fBpPpK0/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[0])**2 + abs(amplitudes_DeltaS1b8X8[0])**2)/GammaBp
    BR_BpP0Kp  = (fBpP0Kp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[1])**2 + abs(amplitudes_DeltaS1b8X8[1])**2)/GammaBp
    BR_B0PmKp  = (fB0PmKp/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[3])**2 + abs(amplitudes_DeltaS1b8X8[3])**2)/GammaB0
    BR_B0P0K0  = (fB0P0K0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[4])**2 + abs(amplitudes_DeltaS1b8X8[4])**2)/GammaB0
    BR_B0sK0K0b= (fB0sK0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[6])**2 + abs(amplitudes_DeltaS1b8X8[6])**2)/GammaB0s
    BR_B0sPpPm = (fB0sPpPm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[7])**2 + abs(amplitudes_DeltaS1b8X8[7])**2)/GammaB0s
    BR_B0sKpKm = (fB0sKpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[8])**2 + abs(amplitudes_DeltaS1b8X8[8])**2)/GammaB0s
    BR_B0sP0P0 = (fB0sP0P0/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[9])**2 + abs(amplitudes_DeltaS1b8X8[9])**2)/GammaB0s

    #===Direct CP asymmetries Delta S = 1===#
    ACP_BpPpK0  = (abs(amplitudes_DeltaS1b8X8[0])**2 - abs(amplitudes_DeltaS18X8[0])**2)/(abs(amplitudes_DeltaS1b8X8[0])**2 + abs(amplitudes_DeltaS18X8[0])**2)
    ACP_BpP0Kp  = (abs(amplitudes_DeltaS1b8X8[1])**2 - abs(amplitudes_DeltaS18X8[1])**2)/(abs(amplitudes_DeltaS1b8X8[1])**2 + abs(amplitudes_DeltaS18X8[1])**2)
    ACP_B0PmKp  = (abs(amplitudes_DeltaS1b8X8[3])**2 - abs(amplitudes_DeltaS18X8[3])**2)/(abs(amplitudes_DeltaS1b8X8[3])**2 + abs(amplitudes_DeltaS18X8[3])**2)
    ACP_B0P0K0  = (abs(amplitudes_DeltaS1b8X8[4])**2 - abs(amplitudes_DeltaS18X8[4])**2)/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    ACP_B0sK0K0b= (abs(amplitudes_DeltaS1b8X8[6])**2 - abs(amplitudes_DeltaS18X8[6])**2)/(abs(amplitudes_DeltaS1b8X8[6])**2 + abs(amplitudes_DeltaS18X8[6])**2)
    ACP_B0sPpPm = (abs(amplitudes_DeltaS1b8X8[7])**2 - abs(amplitudes_DeltaS18X8[7])**2)/(abs(amplitudes_DeltaS1b8X8[7])**2 + abs(amplitudes_DeltaS18X8[7])**2)
    ACP_B0sKpKm = (abs(amplitudes_DeltaS1b8X8[8])**2 - abs(amplitudes_DeltaS18X8[8])**2)/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    ACP_B0sP0P0 = (abs(amplitudes_DeltaS1b8X8[9])**2 - abs(amplitudes_DeltaS18X8[9])**2)/(abs(amplitudes_DeltaS1b8X8[9])**2 + abs(amplitudes_DeltaS18X8[9])**2)


    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag

    #===Chi2 Contributions===#
    #===Branching ratios===#
    chi2_BR_BpPpK0  = (BR_BpPpK0 - BpPpK0_exp)**2/(BpPpK0_inc**2)
    chi2_BR_BpP0Kp  = (BR_BpP0Kp - BpP0Kp_exp)**2/(BpP0Kp_inc**2)
    chi2_BR_B0PmKp  = (BR_B0PmKp - B0PmKp_exp)**2/(B0PmKp_inc**2)
    chi2_BR_B0P0K0  = (BR_B0P0K0 - B0P0K0_exp)**2/(B0P0K0_inc**2)
    chi2_BR_B0sK0K0b= (BR_B0sK0K0b - B0sK0K0b_exp)**2/(B0sK0K0b_inc**2)
    chi2_BR_B0sPpPm = (BR_B0sPpPm - B0sPpPm_exp)**2/(B0sPpPm_inc**2)
    chi2_BR_B0sKpKm = (BR_B0sKpKm - B0sKpKm_exp)**2/(B0sKpKm_inc**2)
    chi2_BR_B0sP0P0 = (BR_B0sP0P0 - B0sP0P0_exp)**2/(B0sP0P0_inc**2)

    #===Direct CP asymmetries===#
    chi2_ACP_BpPpK0  = (ACP_BpPpK0 - ACP_BpPpK0_exp)**2/(ACP_BpPpK0_inc**2)
    chi2_ACP_BpP0Kp  = (ACP_BpP0Kp - ACP_BpP0Kp_exp)**2/(ACP_BpP0Kp_inc**2)
    chi2_ACP_B0PmKp  = (ACP_B0PmKp - ACP_B0PmKp_exp)**2/(ACP_B0PmKp_inc**2)
    chi2_ACP_B0P0K0  = (ACP_B0P0K0 - ACP_B0P0K0_exp)**2/(ACP_B0P0K0_inc**2)
    #chi2_ACP_B0sK0K0b= (ACP_B0sK0K0b - ACP_B0sK0K0b_exp)**2/(ACP_B0sK0K0b_inc**2)
    #chi2_ACP_B0sPpPm = (ACP_B0sPpPm - ACP_B0sPpPm_exp)**2/(ACP_B0sPpPm_inc**2)
    chi2_ACP_B0sKpKm = (ACP_B0sKpKm - ACP_B0sKpKm_exp)**2/(ACP_B0sKpKm_inc**2)
    #chi2_ACP_B0sP0P0 = (ACP_B0sP0P0 - ACP_B0sP0P0_exp)**2/(ACP_B0sP0P0_inc**2)

    #===Indirect CP asymmetries===#
    chi2_SCP_B0P0K0     = (Im_SCP_B0P0K0 - SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc**2)
    chi2_SCP_B0sKpKm    = (Im_SCP_B0sKpKm - SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc**2)

    chi2_BR = chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0sKpKm

    chi2_SCP = chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm 

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total


def chi2MinuitPrint(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    print("chi2 BR BpK0bKp = ",chi2_BR_BpK0bKp)
    print("chi2 BR BpP0Pp = ",chi2_BR_BpP0Pp)
    print("chi2 BR BpEPp = ",chi2_BR_BpEPp)
    print("chi2 BR BpEpPp = ",chi2_BR_BpEpPp)
    print("chi2 BR B0K0K0b = ",chi2_BR_B0K0K0b)
    print("chi2 BR B0PpPm = ",chi2_BR_B0PpPm)
    print("chi2 BR B0KpKm = ",chi2_BR_B0KpKm)
    print("chi2 BR B0P0P0 = ",chi2_BR_B0P0P0)
    print("chi2 BR B0P0E = ",chi2_BR_B0P0E)
    print("chi2 BR B0P0Ep = ",chi2_BR_B0P0Ep)
    print("chi2 BR B0EE = ",chi2_BR_B0EE)
    print("chi2 BR B0EpEp = ",chi2_BR_B0EpEp)
    print("chi2 BR B0sPpKm = ",chi2_BR_B0sPpKm)
    print("chi2 BR BpPpK0 = ",chi2_BR_BpPpK0)
    print("chi2 BR BpP0Kp = ",chi2_BR_BpP0Kp)
    print("chi2 BR BpEKp = ",chi2_BR_BpEKp)
    print("chi2 BR BpEpKp = ",chi2_BR_BpEpKp)
    print("chi2 BR B0PmKp = ",chi2_BR_B0PmKp)
    print("chi2 BR B0P0K0 = ",chi2_BR_B0P0K0)
    print("chi2 BR B0EK0 = ",chi2_BR_B0EK0)
    print("chi2 BR B0EpK0 = ",chi2_BR_B0EpK0)
    print("chi2 BR B0sK0K0b = ",chi2_BR_B0sK0K0b)
    print("chi2 BR B0sPpPm = ",chi2_BR_B0sPpPm)
    print("chi2 BR B0sKpKm = ",chi2_BR_B0sKpKm)
    print("chi2 BR B0sP0P0 = ",chi2_BR_B0sP0P0)
    print("chi2 BR B0sEE = ",chi2_BR_B0sEE)
    print("chi2 BR B0sEEp = ",chi2_BR_B0sEEp)
    print("chi2 BR B0sEpEp = ",chi2_BR_B0sEpEp)
    print("chi2 ACP BpK0bKp = ",chi2_ACP_BpK0bKp)
    print("chi2 ACP BpP0Pp = ",chi2_ACP_BpP0Pp)
    print("chi2 ACP BpEPp = ",chi2_ACP_BpEPp)
    print("chi2 ACP BpEpPp = ",chi2_ACP_BpEpPp)
    print("chi2 ACP B0K0K0b = ",chi2_ACP_B0K0K0b)
    print("chi2 ACP B0PpPm = ",chi2_ACP_B0PpPm)
    print("chi2 ACP B0P0P0 = ",chi2_ACP_B0P0P0)
    print("chi2 ACP B0sPpKm = ",chi2_ACP_B0sPpKm)
    print("chi2 ACP BpPpK0 = ",chi2_ACP_BpPpK0)
    print("chi2 ACP BpP0Kp = ",chi2_ACP_BpP0Kp)
    print("chi2 ACP BpEKp = ",chi2_ACP_BpEKp)
    print("chi2 ACP BpEpKp = ",chi2_ACP_BpEpKp)
    print("chi2 ACP B0PmKp = ",chi2_ACP_B0PmKp)
    print("chi2 ACP B0P0K0 = ",chi2_ACP_B0P0K0)
    print("chi2 ACP B0EpK0 = ",chi2_ACP_B0EpK0)
    print("chi2 ACP B0sKpKm = ",chi2_ACP_B0sKpKm)
    print("chi2 SCP B0K0K0b = ",chi2_SCP_B0K0K0b)
    print("chi2 SCP B0PpPm = ",chi2_SCP_B0PpPm)
    print("chi2 SCP B0P0K0 = ",chi2_SCP_B0P0K0)
    print("chi2 SCP B0sKpKm = ",chi2_SCP_B0sKpKm)
    print("chi2 SCP B0EpK0 = ",chi2_SCP_B0EpK0)
    print()
    print("BR B0EpK0 =",BR_B0EpK0)


    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total


def chi2MinuitNoEtaEta(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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


    #===Calculating amplitudes===#
    amplitudes_DeltaS08X8 = np.matmul(DIAGd08X8,diagramsd08X8)
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS0b8X8= np.matmul(DIAGd08X8,diagramsd0b8X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    amplitudes_DeltaS08X1 = np.matmul(DIAGD08X1,diagramsd08X1)
    amplitudes_DeltaS18X1 = np.matmul(DIAGD18X1,diagramsd18X1)
    amplitudes_DeltaS0b8X1= np.matmul(DIAGD08X1,diagramsd0b8X1)
    amplitudes_DeltaS1b8X1= np.matmul(DIAGD18X1,diagramsd1b8X1)


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


    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_BpEPp + chi2_BR_BpEpPp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0P0E + chi2_BR_B0P0Ep + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_BpEKp + chi2_BR_BpEpKp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0EK0 + chi2_BR_B0EpK0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_BpEPp + chi2_ACP_BpEpPp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_BpEKp + chi2_ACP_BpEpKp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0EpK0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm + chi2_SCP_B0EpK0

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total

def chi2MinuitNoEtaEtaDS1only(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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

    #===EWP Diagrams===#
    PEWTsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8+AA8X8)+(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWCsb8X8 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X8+CC8X8-AA8X8)-(c9-c10)/(c1-c2)*(TT8X8-CC8X8-AA8X8))*Vtbsb
    PEWAsb8X8 = -3/2 * (c9+c10)/(c1+c2)*AA8X8*Vtbsb

    PEWTsb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)+(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbsb
    PEWCsb8X1 = -3/4 * ((c9+c10)/(c1+c2)*(TT8X1+CC8X1)-(c9-c10)/(c1-c2)*(TT8X1-CC8X1))*Vtbsb

    #===Diagram contributions to amplitudes===#

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
    DIAGD18X1 = [[1/root3,1/root3,2/root3,2/root3,0,1/(3*root3)],\
                [0,1/root3,2/root3,2/root3,0,-2/(3*root3)],\
                [0,-1/root6,0,0,-1/root6,0],\
                [0,1/(3*root2),2*root2/3,2*root2/3,-1/(3*root2),-2*root2/9]]
    DIAGD11X1 = [[root2/3,root2/3]]


    #===Vector of diagrams===#
    diagramsd18X8 = np.array([TTs8X8,CCs8X8,PPucs8X8,AAs8X8,PAucs8X8,PPtcs8X8,PAtcs8X8,PEWTs8X8,PEWCs8X8,PEWAs8X8])
    diagramsd1b8X8= np.array([TTsb8X8,CCsb8X8,PPucsb8X8,AAsb8X8,PAucsb8X8,PPtcsb8X8,PAtcsb8X8,PEWTsb8X8,PEWCsb8X8,PEWAsb8X8])
    
    diagramsd18X1 = np.array([TTs8X1,CCs8X1,PPucs8X1,PPtcs8X1,PEWTs8X1,PEWCs8X1])
    diagramsd1b8X1= np.array([TTsb8X1,CCsb8X1,PPucsb8X1,PPtcsb8X1,PEWTsb8X1,PEWCsb8X1])


    #===Amplitudes===#
    #===Initialising amplitudes===#
    amplitudes_DeltaS18X8  = np.zeros(12)
    amplitudes_DeltaS1b8X8 = np.zeros(12)

    amplitudes_DeltaS18X1 = np.zeros(6)
    amplitudes_DeltaS1b8X1= np.zeros(6)


    #===Calculating amplitudes===#
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    amplitudes_DeltaS18X1 = np.matmul(DIAGD18X1,diagramsd18X1)
    amplitudes_DeltaS1b8X1= np.matmul(DIAGD18X1,diagramsd1b8X1)


    #===Calculating branching ratios===#
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

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
    Im_SCP_B0EpK0  = SCP_B0EpK0.imag

    #===Chi2 Contributions===#
    #===Branching ratios===#
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

    #===Direct CP asymmetries===#
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
    chi2_SCP_B0P0K0     = (Im_SCP_B0P0K0 - SCP_B0P0K0_exp)**2/(SCP_B0P0K0_inc**2)
    chi2_SCP_B0sKpKm    = (Im_SCP_B0sKpKm - SCP_B0sKpKm_exp)**2/(SCP_B0sKpKm_inc**2)
    chi2_SCP_B0EpK0     = (Im_SCP_B0EpK0 - SCP_B0EpK0_exp)**2/(SCP_B0EpK0_inc**2)

    chi2_BR = chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_BpEKp + chi2_BR_BpEpKp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0EK0 + chi2_BR_B0EpK0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_BpEKp + chi2_ACP_BpEpKp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0EpK0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm + chi2_SCP_B0EpK0

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total


def chi2MinuitNoEtaEtaToverCfixed(ampT8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    ToverCratio = 0.2
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    CC8X8 = ToverCratio*abs(ampT8X8)*np.exp(1j*delC8X8)
    PPuc8X8= abs(ampPuc8X8)*np.exp(1j*delPuc8X8)
    AA8X8 = abs(ampA8X8)*np.exp(1j*delA8X8)
    PAuc8X8 = abs(ampPAuc8X8)*np.exp(1j*delPAuc8X8)
    PPtc8X8 = abs(ampPtc8X8)*np.exp(1j*delPtc8X8)
    PAtc8X8 = abs(ampPAtc8X8)*np.exp(1j*delPAtc8X8)

    TT8X1 = abs(ampT8X1)*np.exp(1j*delT8X1)
    CC8X1 = ToverCratio*abs(ampT8X1)*np.exp(1j*delC8X1)
    PPuc8X1= abs(ampPuc8X1)*np.exp(1j*delPuc8X1)
    PPtc8X1 = abs(ampPtc8X1)*np.exp(1j*delPtc8X1)
    
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


    #===Calculating amplitudes===#
    amplitudes_DeltaS08X8 = np.matmul(DIAGd08X8,diagramsd08X8)
    amplitudes_DeltaS18X8 = np.matmul(DIAGd18X8,diagramsd18X8)
    amplitudes_DeltaS0b8X8= np.matmul(DIAGd08X8,diagramsd0b8X8)
    amplitudes_DeltaS1b8X8= np.matmul(DIAGd18X8,diagramsd1b8X8)

    amplitudes_DeltaS08X1 = np.matmul(DIAGD08X1,diagramsd08X1)
    amplitudes_DeltaS18X1 = np.matmul(DIAGD18X1,diagramsd18X1)
    amplitudes_DeltaS0b8X1= np.matmul(DIAGD08X1,diagramsd0b8X1)
    amplitudes_DeltaS1b8X1= np.matmul(DIAGD18X1,diagramsd1b8X1)


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


    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    chi2_BR = chi2_BR_BpK0bKp + chi2_BR_BpP0Pp + chi2_BR_BpEPp + chi2_BR_BpEpPp + chi2_BR_B0K0K0b + chi2_BR_B0PpPm + chi2_BR_B0KpKm + chi2_BR_B0P0P0 + chi2_BR_B0P0E + chi2_BR_B0P0Ep + chi2_BR_B0sPpKm +\
              chi2_BR_BpPpK0 + chi2_BR_BpP0Kp + chi2_BR_BpEKp + chi2_BR_BpEpKp + chi2_BR_B0PmKp + chi2_BR_B0P0K0 + chi2_BR_B0EK0 + chi2_BR_B0EpK0 + chi2_BR_B0sK0K0b + chi2_BR_B0sPpPm + chi2_BR_B0sKpKm + chi2_BR_B0sP0P0
    
    chi2_ACP = chi2_ACP_BpK0bKp + chi2_ACP_BpP0Pp + chi2_ACP_BpEPp + chi2_ACP_BpEpPp + chi2_ACP_B0K0K0b + chi2_ACP_B0PpPm + chi2_ACP_B0P0P0 + chi2_ACP_B0sPpKm +\
               chi2_ACP_BpPpK0 + chi2_ACP_BpP0Kp + chi2_ACP_BpEKp + chi2_ACP_BpEpKp + chi2_ACP_B0PmKp + chi2_ACP_B0P0K0 + chi2_ACP_B0EpK0 + chi2_ACP_B0sKpKm
    chi2_SCP = chi2_SCP_B0K0K0b + chi2_SCP_B0PpPm + chi2_SCP_B0P0K0 + chi2_SCP_B0sKpKm + chi2_SCP_B0EpK0

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP
    return chi2_total

def chi2MinuitWithTheta(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1,theta_eta):
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    BR_BpEPp   = (fBpEPp/(16*math.pi*mBp**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2)/GammaBp
    BR_BpEpPp  = (fBpEpPp/(16*math.pi*mBp**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2)/GammaBp
    BR_B0K0K0b = (fB0K0K0b/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[3])**2 +abs(amplitudes_DeltaS0b8X8[3])**2)/GammaB0
    BR_B0PpPm  = (fB0PpPm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[4])**2 + abs(amplitudes_DeltaS0b8X8[4])**2)/GammaB0
    BR_B0KpKm  = (fB0KpKm/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[5])**2 + abs(amplitudes_DeltaS0b8X8[5])**2)/GammaB0
    BR_B0P0P0  = (fB0P0P0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS08X8[6])**2 + abs(amplitudes_DeltaS0b8X8[6])**2)/GammaB0
    BR_B0P0E   = (fB0P0E/(16*math.pi*mB0**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2)/GammaB0
    BR_B0P0Ep  = (fB0P0Ep/(16*math.pi*mB0**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2)/GammaB0
    BR_B0sPpKm = (fB0sPpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[9])**2 + abs(amplitudes_DeltaS0b8X8[9])**2)/GammaB0s
    BR_B0sP0K0b= (fB0sP0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS08X8[10])**2 + abs(amplitudes_DeltaS0b8X8[10])**2)/GammaB0s
    BR_B0sEK0b = (fB0sEK0b/(16*math.pi*mB0s**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2)/GammaB0s
    BR_B0sEpK0b= (fB0sEpK0b/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2)/GammaB0s
    
    BR_B0EE    = (fB0EE/(16*math.pi*mB0**2))*((abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta)))**2 + abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2)/GammaB0
    BR_B0EEp   = (fB0EEp/(16*math.pi*mB0**2))*((abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta)))**2 + abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2)/GammaB0
    BR_B0EpEp  = (fB0EpEp/(16*math.pi*mB0**2))*((abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta)))**2 + abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2)/GammaB0

    #===Branching ratios Delta S = 1===#
    BR_BpPpK0  = (fBpPpK0/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[0])**2 + abs(amplitudes_DeltaS1b8X8[0])**2)/GammaBp
    BR_BpP0Kp  = (fBpP0Kp/(16*math.pi*mBp**2))*(abs(amplitudes_DeltaS18X8[1])**2 + abs(amplitudes_DeltaS1b8X8[1])**2)/GammaBp
    BR_BpEKp   = (fBpEKp/(16*math.pi*mBp**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2)/GammaBp
    BR_BpEpKp  = (fBpEpKp/(16*math.pi*mBp**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2)/GammaBp
    BR_B0PmKp  = (fB0PmKp/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[3])**2 + abs(amplitudes_DeltaS1b8X8[3])**2)/GammaB0
    BR_B0P0K0  = (fB0P0K0/(16*math.pi*mB0**2))*(abs(amplitudes_DeltaS18X8[4])**2 + abs(amplitudes_DeltaS1b8X8[4])**2)/GammaB0
    BR_B0EK0   = (fB0EK0/(16*math.pi*mB0**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2)/GammaB0
    BR_B0EpK0  = (fB0EpK0/(16*math.pi*mB0**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2)/GammaB0
    BR_B0sK0K0b= (fB0sK0K0b/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[6])**2 + abs(amplitudes_DeltaS1b8X8[6])**2)/GammaB0s
    BR_B0sPpPm = (fB0sPpPm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[7])**2 + abs(amplitudes_DeltaS1b8X8[7])**2)/GammaB0s
    BR_B0sKpKm = (fB0sKpKm/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[8])**2 + abs(amplitudes_DeltaS1b8X8[8])**2)/GammaB0s
    BR_B0sP0P0 = (fB0sP0P0/(16*math.pi*mB0s**2))*(abs(amplitudes_DeltaS18X8[9])**2 + abs(amplitudes_DeltaS1b8X8[9])**2)/GammaB0s
    BR_B0sP0E  = (fB0sP0E/(16*math.pi*mB0s**2))*(abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2)/GammaB0s
    BR_B0sP0Ep = (fB0sP0Ep/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2)/GammaB0s

    BR_B0sEE   = (fB0sEE/(16*math.pi*mB0s**2))*(abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2 + abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2)/GammaB0s
    BR_B0sEEp  = (fB0sEEp/(16*math.pi*mB0s**2))*(abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2 + abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2)/GammaB0s
    BR_B0sEpEp = (fB0sEpEp/(16*math.pi*mB0s**2))*(abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2 + abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2)/GammaB0s

    #===Direct CP asymmetries Delta S = 0===#
    ACP_BpK0bKp = (abs(amplitudes_DeltaS0b8X8[0])**2 - abs(amplitudes_DeltaS08X8[0])**2)/(abs(amplitudes_DeltaS0b8X8[0])**2 + abs(amplitudes_DeltaS08X8[0])**2)
    ACP_BpP0Pp  = (abs(amplitudes_DeltaS0b8X8[1])**2 - abs(amplitudes_DeltaS08X8[1])**2)/(abs(amplitudes_DeltaS0b8X8[1])**2 + abs(amplitudes_DeltaS08X8[1])**2)
    ACP_BpEPp   = (abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2)
    ACP_BpEpPp  = (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[2],amplitudes_DeltaS0b8X1[0],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[2],amplitudes_DeltaS08X1[0],theta_eta))**2)
    ACP_B0K0K0b = (abs(amplitudes_DeltaS0b8X8[3])**2 - abs(amplitudes_DeltaS08X8[3]**2))/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    ACP_B0PpPm  = (abs(amplitudes_DeltaS0b8X8[4])**2 - abs(amplitudes_DeltaS08X8[4])**2)/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    ACP_B0KpKm  = (abs(amplitudes_DeltaS0b8X8[5])**2 - abs(amplitudes_DeltaS08X8[5])**2)/(abs(amplitudes_DeltaS0b8X8[5])**2 + abs(amplitudes_DeltaS08X8[5])**2)
    ACP_B0P0P0  = (abs(amplitudes_DeltaS0b8X8[6])**2 - abs(amplitudes_DeltaS08X8[6])**2)/(abs(amplitudes_DeltaS0b8X8[6])**2 + abs(amplitudes_DeltaS08X8[6])**2)
    ACP_B0P0E   = (abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2)
    ACP_B0P0Ep  = (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[7],amplitudes_DeltaS0b8X1[1],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[7],amplitudes_DeltaS08X1[1],theta_eta))**2)
    ACP_B0sPpKm = (abs(amplitudes_DeltaS0b8X8[9])**2 - abs(amplitudes_DeltaS08X8[9])**2)/(abs(amplitudes_DeltaS0b8X8[9])**2 + abs(amplitudes_DeltaS08X8[9])**2)
    ACP_B0sP0K0b= (abs(amplitudes_DeltaS0b8X8[10])**2 - abs(amplitudes_DeltaS08X8[10])**2)/(abs(amplitudes_DeltaS0b8X8[10])**2 + abs(amplitudes_DeltaS08X8[10])**2)
    ACP_B0sEK0b = (abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2)
    ACP_B0sEpK0b= (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS0b8X8[11],amplitudes_DeltaS0b8X1[3],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS08X8[11],amplitudes_DeltaS08X1[3],theta_eta))**2)

    ACP_B0EE    = (abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 - abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)/(abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 + abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)
    ACP_B0EEp   = (abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 - abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)/(abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 + abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)
    ACP_B0EpEp  = (abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 - abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)/(abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS0b8X8[8],amplitudes_DeltaS0b8X1[2],amplitudes_DeltaS0b1X1[0],theta_eta))**2 + abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS08X8[8],amplitudes_DeltaS08X1[2],amplitudes_DeltaS01X1[0],theta_eta))**2)
    
    #===Direct CP asymmetries Delta S = 1===#
    ACP_BpPpK0  = (abs(amplitudes_DeltaS1b8X8[0])**2 - abs(amplitudes_DeltaS18X8[0])**2)/(abs(amplitudes_DeltaS1b8X8[0])**2 + abs(amplitudes_DeltaS18X8[0])**2)
    ACP_BpP0Kp  = (abs(amplitudes_DeltaS1b8X8[1])**2 - abs(amplitudes_DeltaS18X8[1])**2)/(abs(amplitudes_DeltaS1b8X8[1])**2 + abs(amplitudes_DeltaS18X8[1])**2)
    ACP_BpEKp   = (abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2)
    ACP_BpEpKp  = (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[2],amplitudes_DeltaS1b8X1[0],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[2],amplitudes_DeltaS18X1[0],theta_eta))**2)
    ACP_B0PmKp  = (abs(amplitudes_DeltaS1b8X8[3])**2 - abs(amplitudes_DeltaS18X8[3])**2)/(abs(amplitudes_DeltaS1b8X8[3])**2 + abs(amplitudes_DeltaS18X8[3])**2)
    ACP_B0P0K0  = (abs(amplitudes_DeltaS1b8X8[4])**2 - abs(amplitudes_DeltaS18X8[4])**2)/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    ACP_B0EK0   = (abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2)
    ACP_B0EpK0  = (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2)
    ACP_B0sK0K0b= (abs(amplitudes_DeltaS1b8X8[6])**2 - abs(amplitudes_DeltaS18X8[6])**2)/(abs(amplitudes_DeltaS1b8X8[6])**2 + abs(amplitudes_DeltaS18X8[6])**2)
    ACP_B0sPpPm = (abs(amplitudes_DeltaS1b8X8[7])**2 - abs(amplitudes_DeltaS18X8[7])**2)/(abs(amplitudes_DeltaS1b8X8[7])**2 + abs(amplitudes_DeltaS18X8[7])**2)
    ACP_B0sKpKm = (abs(amplitudes_DeltaS1b8X8[8])**2 - abs(amplitudes_DeltaS18X8[8])**2)/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    ACP_B0sP0P0 = (abs(amplitudes_DeltaS1b8X8[9])**2 - abs(amplitudes_DeltaS18X8[9])**2)/(abs(amplitudes_DeltaS1b8X8[9])**2 + abs(amplitudes_DeltaS18X8[9])**2)
    ACP_B0sP0E  = (abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2 - abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2)/(abs(amplitude_eta_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2 + abs(amplitude_eta_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2)
    ACP_B0sP0Ep = (abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2 - abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2)/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[10],amplitudes_DeltaS1b8X1[2],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[10],amplitudes_DeltaS18X1[2],theta_eta))**2)

    ACP_B0sEE   = (abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 - abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)/(abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 + abs(amplitude_eta_eta_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)
    ACP_B0sEEp  = (abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 - abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)/(abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 + abs(amplitude_eta_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)
    ACP_B0sEpEp = (abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 - abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)/(abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS1b8X8[11],amplitudes_DeltaS1b8X1[3],amplitudes_DeltaS1b1X1[0],theta_eta))**2 + abs(amplitude_eta_prime_eta_prime_with_theta(amplitudes_DeltaS18X8[11],amplitudes_DeltaS18X1[3],amplitudes_DeltaS11X1[0],theta_eta))**2)

    #===Indirect CP asymmetries Delta S = 0===#
    SCP_B0K0K0b    = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[3])*amplitudes_DeltaS0b8X8[3])/(abs(amplitudes_DeltaS0b8X8[3])**2 + abs(amplitudes_DeltaS08X8[3])**2)
    Im_SCP_B0K0K0b = SCP_B0K0K0b.imag
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))*amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))/(abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1],theta_eta))**2 + abs(amplitude_eta_prime_with_theta(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1],theta_eta))**2)
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
    return chi2_total

def chi2MinuitAsmall(ampT8X8,ampC8X8,ampPuc8X8,ampA8X8,ampPAuc8X8,ampPtc8X8,ampPAtc8X8,delC8X8,delPuc8X8,delA8X8,delPAuc8X8,delPtc8X8,delPAtc8X8,ampT8X1,ampC8X1,ampPuc8X1,ampPtc8X1,delT8X1,delC8X1,delPuc8X1,delPtc8X1,ampC1X1,ampPtc1X1,delC1X1,delPtc1X1):
    weight = 1e7
    root2 = np.sqrt(2)
    root3 = np.sqrt(3)
    root6 = np.sqrt(6)
    Vud,Vus,Vub,Vtd,Vts,Vtb,gamma,beta,betas = Vud_exp,Vus_exp,Vub_exp,Vtd_exp,Vts_exp,Vtb_exp,gamma_exp,beta_exp,betas_exp

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
    SCP_B0PpPm     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS08X8[4])*amplitudes_DeltaS0b8X8[4])/(abs(amplitudes_DeltaS0b8X8[4])**2 + abs(amplitudes_DeltaS08X8[4])**2)
    Im_SCP_B0PpPm  = SCP_B0PpPm.imag

    #===Indirect CP asymmetries Delta S = 1===#
    SCP_B0P0K0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitudes_DeltaS18X8[4])*amplitudes_DeltaS1b8X8[4])/(abs(amplitudes_DeltaS1b8X8[4])**2 + abs(amplitudes_DeltaS18X8[4])**2)
    Im_SCP_B0P0K0  = SCP_B0P0K0.imag
    SCP_B0sKpKm    = 2*(np.exp(-2j*betas)*np.conjugate(amplitudes_DeltaS18X8[8])*amplitudes_DeltaS1b8X8[8])/(abs(amplitudes_DeltaS1b8X8[8])**2 + abs(amplitudes_DeltaS18X8[8])**2)
    Im_SCP_B0sKpKm = SCP_B0sKpKm.imag
    SCP_B0EpK0     = 2*(np.exp(-2j*beta)*np.conjugate(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))*amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))/(abs(amplitude_eta_prime(amplitudes_DeltaS1b8X8[5],amplitudes_DeltaS1b8X1[1]))**2 + abs(amplitude_eta_prime(amplitudes_DeltaS18X8[5],amplitudes_DeltaS18X1[1]))**2)
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

    chi2_total = chi2_BR + chi2_ACP + chi2_SCP + min(weight*(abs(ampA8X8)-0.2*abs(ampT8X8)),0)**2
    return chi2_total


