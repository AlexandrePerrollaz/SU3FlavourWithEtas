import numpy as np
from iminuit import Minuit
from scipy import stats
import random as rd
from chi2_functions import *
from exp_data import *
from tqdm import tqdm

params = []
results = []
parameters = []
chi2results = []
good_fits = []
count = 0
error = []

n = 1 # Number of attemps
for i in tqdm(range(n),leave = True):
    rand_T, rand_C, rand_Puc, rand_A, rand_PAuc, rand_Ptc, rand_PAtc = rd.random()*10,rd.random()*10,rd.random()*10,rd.random()*10,rd.random()*10,rd.random()*10,rd.random()*10
    rand_delC,rand_delP,rand_delA,rand_delPA,rand_delP2,rand_delPA2 = rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi
    rand_T8X1, rand_C8X1, rand_Puc8X1, rand_Ptc8X1 = rd.random()*10,rd.random()*10,rd.random()*10,rd.random()*10
    rand_delT8X1,rand_delC8X1,rand_delPuc8X1,rand_delPtc8X1 = rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi,rd.random()*2*np.pi
    rand_C1X1,rand_Ptc1X1 = rd.random()*10,rd.random()*10
    rand_delC1X1,rand_delPtc1X1 = rd.random()*2*np.pi,rd.random()*2*np.pi

    m = Minuit(chi2Minuit,ampT8X8=rand_T,ampC8X8=rand_C,ampPuc8X8=rand_Puc,ampA8X8=rand_A,ampPAuc8X8=rand_PAuc,ampPtc8X8=rand_Ptc,ampPAtc8X8=rand_PAtc,\
               delC8X8=rand_delC,delPuc8X8=rand_delP,delA8X8=rand_delA,delPAuc8X8 = rand_delPA,delPtc8X8=rand_delP2,delPAtc8X8=rand_delPA2,\
               ampT8X1=rand_T8X1,ampC8X1=rand_C8X1,ampPuc8X1=rand_Puc8X1,ampPtc8X1=rand_Ptc8X1,\
               delT8X1=rand_delT8X1,delC8X1=rand_delC8X1,delPuc8X1=rand_delPuc8X1,delPtc8X1=rand_delPtc8X1,\
               ampC1X1=rand_C1X1,ampPtc1X1=rand_Ptc1X1,\
               delC1X1=rand_delC1X1,delPtc1X1=rand_delPtc1X1)


    #============ Fixing the CKM matrix ============#
    m.limits["ampT8X8"] = (0,None)
    m.limits["ampC8X8"] = (0,None)
    m.limits["ampPuc8X8"] = (0,None)
    m.limits["ampA8X8"] = (0,None)
    m.limits["ampPAuc8X8"] = (0,None)
    m.limits["ampPtc8X8"] = (0,None)
    m.limits["ampPAtc8X8"] = (0,None)

    m.limits["ampT8X1"] = (0,None)
    m.limits["ampC8X1"] = (0,None)
    m.limits["ampPuc8X1"] = (0,None)
    m.limits["ampPtc8X1"] = (0,None)

    m.limits["ampC1X1"] = (0,None)
    m.limits["ampPtc1X1"] = (0,None)

    m.limits["delC8X8"] = (0,2*np.pi)
    m.limits["delPuc8X8"] = (0,2*np.pi)
    m.limits["delA8X8"] = (0,2*np.pi)
    m.limits["delPAuc8X8"] = (0,2*np.pi)
    m.limits["delPtc8X8"] = (0,2*np.pi)
    m.limits["delPAtc8X8"] = (0,2*np.pi)

    m.limits["delT8X1"] = (0,2*np.pi)
    m.limits["delC8X1"] = (0,2*np.pi)
    m.limits["delPuc8X1"] = (0,2*np.pi)
    m.limits["delPtc8X1"] = (0,2*np.pi)

    m.limits["delC1X1"] = (0,2*np.pi)
    m.limits["delPtc1X1"] = (0,2*np.pi)

    m.migrad()
    results.append(m.fval)
    parameters.append(m.values)
    par = np.array(m.values)



res_array = np.array(results)
ind_min = res_array.argmin()
params = np.array(parameters[ind_min])
print()
print("===========Results==========")
print("Minimum fval =",res_array[ind_min])
print("p_value =",stats.chi2.sf(res_array[ind_min], df=5))
print("Parameters = ",np.array(parameters[ind_min]))

print("p-value =",stats.chi2.sf(57,df = 23))