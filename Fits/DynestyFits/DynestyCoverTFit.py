import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from multiprocessing import Pool
import dynesty
from dynesty import plotting as dyplot
import dynesty.utils as dyut
from DynestyChi2Functions import *
import pickle

def prior_transform(u):
  u_ampT8X8, u_ampPuc8X8, u_ampA8X8, u_ampPAuc8X8, u_ampPtc8X8, u_ampPAtc8X8,\
  u_delC8X8, u_delPuc8X8, u_delA8X8, u_delPAuc8X8, u_delPtc8X8, u_delPAtc8X8,\
  u_ampT8X1, u_ampPuc8X1, u_ampPtc8X1,\
  u_delT8X1, u_delC8X1, u_delPuc8X1, u_delPtc8X1 = u

  ampT8X8 = 100*u_ampT8X8
  ampPuc8X8 = 100*u_ampPuc8X8
  ampA8X8 = 100*u_ampA8X8
  ampPAuc8X8 = 100*u_ampPAuc8X8
  ampPtc8X8 = 100*u_ampPtc8X8
  ampPAtc8X8 = 100*u_ampPAtc8X8

  delC8X8 = 2*np.pi*u_delC8X8
  delPuc8X8 = 2*np.pi*u_delPuc8X8
  delA8X8 = 2*np.pi*u_delA8X8
  delPAuc8X8 = 2*np.pi*u_delPAuc8X8
  delPtc8X8 = 2*np.pi*u_delPtc8X8
  delPAtc8X8 = 2*np.pi*u_delPAtc8X8

  ampT8X1 = 100*u_ampT8X1
  ampPuc8X1 = 100*u_ampPuc8X1
  ampPtc8X1 = 100*u_ampPtc8X1

  delT8X1 = 2*np.pi*u_delT8X1
  delC8X1 = 2*np.pi*u_delC8X1
  delPuc8X1 = 2*np.pi*u_delPuc8X1
  delPtc8X1 = 2*np.pi*u_delPtc8X1


  return np.array([ampT8X8, ampPuc8X8, ampA8X8, ampPAuc8X8, ampPtc8X8, ampPAtc8X8,
                   delC8X8, delPuc8X8, delA8X8, delPAuc8X8, delPtc8X8, delPAtc8X8,
                   ampT8X1, ampPuc8X1, ampPtc8X1,
                   delT8X1, delC8X1, delPuc8X1, delPtc8X1])

if __name__ =="__main__":
  nthreads = os.cpu_count()
  ndim = 19
  nlive = 2000
  with mp.Pool(nthreads) as poo:
      dns = dynesty.DynamicNestedSampler(chi2NoEtaEtaToverCfixed,
                                       prior_transform,
                                       ndim = ndim,
                                       nlive=nlive,
                                       sample='rslice',
                                       pool=poo,
                                       queue_size=nthreads * 2)
      dns.run_nested(n_effective=10000)
      
  res = dns.results
  inds = np.arange(len(res.samples))
  inds = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
  samps = res.samples[inds]
  logl  = res.logl[inds]

  dict_result = {
      'dns': dns,
      'samps': samps,
      'logl': logl,
      'logz': res.logz,
      'logzerr': res.logzerr,
  }
  os.makedirs('./BestFits', exist_ok=True)
  with open('./BestFits/Chi2CoverTFit.pkl', 'wb') as f:
    pickle.dump(dict_result, f)
  notification()