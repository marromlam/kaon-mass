# fit_ipatia
#
#

# from ModelBricks import Parameter, Free,  Cat
# from math import sqrt
# from FitLibrary import *
# from AmmModel import *
import numpy as np
# import pycuda.gpuarray as gpuarray
# import pycuda.driver as cudriver
# from tools import plt
# from timeit import default_timer as timer
# import pycuda.cumath
# import pickle as cPickle

# from ipanema import Paramters
import ipanema
import uproot
import complot
import matplotlib.pyplot as plt
import pickle as cPickle
import pandas as pd


__all__ = []
__author__ = ["name"]
__email__ = ["email"]


# Timer0 = timer()

# data = np.float64(cPickle.load(file("./AndreaCuttedtData.crap")))  # Load data
ipanema.initialize('opencl', 1)


mass_range = (440, 580)


# load data {{{

# f = TFile("/scratch03/marcos.romero/K3Pi/ntupleK3pi_MagDown_2016_DTF_1_compact.root")
# t = f.Get("DecayTree")
# h = TH1F("shit", "shit", 10000,440, 580)
# t.Draw("KDTF_M>>shit", "((K_IPCHI2_OWNPV<33.000000)) && ((K_FDCHI2_OWNPV>2.700000)) && ((BDT>-0.030000)) && K_MMERR >1")
# f = ipanema.Sample.from_root("SelectedDataSample2.root", treename="DecayTree")
f = ipanema.Sample.from_root("ntupleK3pi_MagDown_2016_DTF_1_compact.root",
                             branches=['K_IPCHI2_OWNPV', 'K_FDCHI2_OWNPV', 'BDT', 'K_MMERR', 'KDTF_M'], treename="DecayTree")
f.chop("(K_IPCHI2_OWNPV<33.000000) & (K_FDCHI2_OWNPV>2.700000) & (BDT>-0.030000) & K_MMERR >1")
Y, X = np.histogram(f.df['KDTF_M'].values, bins=int(1e5), range=mass_range)
X = ipanema.ristra.allocate(0.5 * (X[1:]+X[:-1]))
Y = ipanema.ristra.allocate(Y)
Z = 0 * Y

# X = np.float64(cPickle.load(open("./AndreaCuttedtData.crap", 'rb')))
# X, Y = np.ascontiguousarray(X[:,0]), np.ascontiguousarray(X[:,1])
# X = ipanema.ristra.allocate(X)
# Y = ipanema.ristra.allocate(Y)
# Z = 0 * Y
# }}}


# create a set of parameters {{{

pars = ipanema.Parameters()

# set species numbers
pars.add(dict(name="npeak", value=6e6, min=0, max=1e13))
pars.add(dict(name="nbkgr", value=6e6, min=0, max=1e13))
# pars.add(dict(name="npeak", value=0.5, min=0, max=1))
# pars.add(dict(name="nbkgr", formula="1-npeak"))

pars.add(dict(name="mu", value=493.467, min=492., max=495.))
pars.add(dict(name="sigma", value=10.1549, min=4., max=35.))
pars.add(dict(name="lanbda", value=-1.048, min=-3, max=-0.5, free=True))
pars.add(dict(name="beta", value=0.0, min=-5e-02, max=5e-02, free=False))
pars.add(dict(name="zeta", value=0.0, min=-5e-02, max=5e-02, free=False))
# pars.add(dict(name="sigma0", value=0.03,      min=0.0000000005, max=3.5))

# fix it to the MC value
pars.add(dict(name="aL", value=0.3744,   free=True, min=0.0,  max=50.2))
pars.add(dict(name="nL", value=9.864,    free=True, min=0.1,  max=55.))
pars.add(dict(name="aR", value=0.964,    free=True, min=0.,   max=55.))
pars.add(dict(name="nR", value=0.5369,   free=True, min=0.1,  max=55.0))
# pars.add(dict(name="m0", value=418.710,  min=416., max=430.))


pars.add(dict(name="a0", value=0.0082947,  min=-0.01, max=.3))
pars.add(dict(name="a1", value=0.2358830,  min=-1.,   max=1.))
pars.add(dict(name="a2", value=-0.2511390, min=-1.,   max=1.))
pars.add(dict(name="a3", value=0.0965313,  min=-1.,   max=1.))
pars.add(dict(name="a4", value=-0.0061410, min=-1.,   max=1.))
pars.add(dict(name="a5", value=0.0052310,  min=-1.,   max=1.))
pars.add(dict(name="a6", value=-0.0100309, min=-1.,   max=1.))
pars.add(dict(name="a7", value=-0.0, min=-1.,   max=1., free=False))
pars.add(dict(name="a8", value=0.0,  min=-1.,   max=1., free=False))
pars.add(dict(name="a9", value=-0.0, min=-1.,   max=1., free=False))


pars.add(dict(name="mLL", value=mass_range[0], free=False))
pars.add(dict(name="mUL", value=mass_range[1], free=False))
pars.add(dict(name="thres", value=0, free=False))

print(pars)

# }}}


prog = ipanema.compile("""

#define USE_DOUBLE 1
#include <lib99ocl/core.c>
#include <lib99ocl/complex.c>
#include <lib99ocl/special.c>
#include <lib99ocl/lineshapes.c>

WITHIN_KERNEL ftype
diego_poly(const ftype x, const ftype a0, const ftype a1, const ftype a2,
           const ftype a3, const ftype a4, const ftype a5, const ftype a6,
           const ftype a7, const ftype a8, const ftype a9)
{
  // x = (self.bins-self.min)/(self.max-self.min)*2.-1
  // print x < -0.99133370127
  // create some arrays
  const ftype x2 = x *x;
  const ftype x3 = x2*x;
  const ftype x4 = x3*x;
  const ftype x5 = x4*x;
  const ftype x6 = x5*x;
  const ftype x7 = x6*x;
  const ftype x8 = x7*x;
  const ftype x9 = x8*x;
  const ftype x10 = x9*x;
  // build the poly
  const ftype pol2  =     0.5 * (    3*x2  - 1.);
  const ftype pol3  =     0.5 * (    5*x3  -      3*x);
  const ftype pol4  =   1./8. * (   35*x4  -     30*x2+3);
  const ftype pol5  =   1./8. * (   63*x5  -     70*x3+15*x);
  const ftype pol6  =  1./16. * (  231*x6  -    315*x4+105*x2-5);
  const ftype pol7  =  1./16. * (  429*x7  -    693*x5+315*x3-35*x);
  const ftype pol8  = 1./128. * ( 6435*x8  -  12012*x6+6930*x4-1260*x2+35);
  const ftype pol9  = 1./128. * (12155*x9  -  25740*x7+18018*x5-4620*x3+315*x);
  const ftype pol10 = 1./256. * (46189*x10 - 109395*x8 + 90090*x6 - 30030*x4 + 3465*x2 - 63);

  ftype ans = 0;
  ans += 1 + a1*x + a2*pol2 + a3*pol3 + a4*pol4 + a5*pol5 + a6*pol6;
  ans += a7*pol7 + a8*pol8 + a9*pol9;
  ans *= exp(a0 * x);
  return ans;
}


KERNEL void
mass_model(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *mass,
           const ftype fpeak, const ftype fbkgr,
           const ftype mu, const ftype sigma, const ftype lanbda, const ftype zeta,
           const ftype beta, const ftype aL, const ftype nL, const ftype aR,
           const ftype nR,
           const ftype a0, const ftype a1, const ftype a2,
           const ftype a3, const ftype a4, const ftype a5, const ftype a6,
           const ftype a7, const ftype a8, const ftype a9,
           const ftype mLL, const ftype mUL, const ftype mThres)
{
  const int idx = get_global_id(0);

  const ftype u_mass = (mass[idx]-mLL)/(mUL-mLL)*2 - 1;
  const ftype peak = ipatia(mass[idx], mu, sigma, lanbda, zeta, beta, aL, nL, aR, nR);
  const ftype bkgr = diego_poly(u_mass, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);

  out[idx] = (mass[idx] > mThres) * (fpeak * peak + fbkgr * bkgr );

}


KERNEL void
mass_peak(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *mass,
          const ftype mu, const ftype sigma, const ftype lanbda, const ftype zeta,
          const ftype beta, const ftype aL, const ftype nL, const ftype aR,
          const ftype nR,
          const ftype mLL, const ftype mUL, const ftype mThres)
{
  const int idx = get_global_id(0);

  const ftype peak = ipatia(mass[idx], mu, sigma, lanbda, zeta, beta, aL, nL, aR, nR);

  out[idx] = (mass[idx] > mThres) ? peak : 0;

}

KERNEL void
mass_bkgr(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *mass,
           const ftype a0, const ftype a1, const ftype a2,
           const ftype a3, const ftype a4, const ftype a5, const ftype a6,
           const ftype a7, const ftype a8, const ftype a9,
           const ftype mLL, const ftype mUL, const ftype mThres)
{
  const int idx = get_global_id(0);

  const ftype u_mass = (mass[idx]-mLL)/(mUL-mLL)*2 - 1;
  const ftype bkgr = diego_poly(u_mass, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);

  out[idx] = (mass[idx] > mThres) ? bkgr : 0;

}


""")


def mass_model(x, y, npeak, nbkgr, mu, sigma, lanbda, zeta, beta, aL, nL, aR, nR,
               a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, mLL, mUL, thres):
    # get signal peak
    _x = ipanema.ristra.allocate(np.linspace(mLL, mUL, 2000))
    _y = 0 * _x
    prog.mass_peak(y, x, np.float64(mu), np.float64(sigma), np.float64(lanbda),
                   np.float64(zeta), np.float64(beta), np.float64(aL), np.float64(nL),
                   np.float64(aR), np.float64(nR), np.float64(mLL),
                   np.float64(mUL), np.float64(thres), global_size=(len(y),))
    peak = ipanema.ristra.get(y)
    peak /= np.sum(peak)
    prog.mass_peak(_y, _x, np.float64(mu), np.float64(sigma), np.float64(lanbda),
                   np.float64(zeta), np.float64(beta), np.float64(aL), np.float64(nL),
                   np.float64(aR), np.float64(nR), np.float64(mLL),
                   np.float64(mUL), np.float64(thres), global_size=(len(_y),))
    ipeak = 1  # np.trapz(ipanema.ristra.get(_y), ipanema.ristra.get(_x))
    # # get background
    prog.mass_bkgr(y, x, np.float64(a0), np.float64(a1), np.float64(a2),
                   np.float64(a3), np.float64(a4), np.float64(a5),
                   np.float64(a6), np.float64(a7), np.float64(a8),
                   np.float64(a9), np.float64(mLL), np.float64(mUL),
                   np.float64(thres), global_size=(len(y),))
    bkgr = ipanema.ristra.get(y)
    bkgr /= np.sum(bkgr)
    prog.mass_bkgr(_y, _x, np.float64(a0), np.float64(a1), np.float64(a2),
                   np.float64(a3), np.float64(a4), np.float64(a5),
                   np.float64(a6), np.float64(a7), np.float64(a8),
                   np.float64(a9), np.float64(mLL), np.float64(mUL),
                   np.float64(thres), global_size=(len(_y),))
    ibkgr = 1  # np.trapz(ipanema.ristra.get(_y), ipanema.ristra.get(_x))
    return npeak * peak/ipeak + nbkgr * bkgr/ibkgr


def fcn(parameters, x, y, z):
    p = parameters.valuesdict()
    prob = mass_model(x, z, **p)
    return -2 * (np.log(prob)*y.get() - prob)


# res = ipanema.optimize(fcn, pars, fcn_args=(X, Y, Z), method='minuit',
#                        tol=50, verbose=True)
# print(res)
# pars = ipanema.Parameters.clone(res.params)
pars.unlock('beta')
res = ipanema.optimize(fcn, pars, fcn_args=(X, Y, Z), method='minuit',
                       tol=1, verbose=False, timeit=True)


# create a nice plot {{{

fig, axplot, axpull = complot.axes_providers.axes_plotpull()
# hdat = complot.hist(f.df['KDTF_M'].values, bins=200, range=mass_range)
# axplot.errorbar(hdat.bins, hdat.counts, yerr=hdat.yerr, fmt='.', color='k')

# take original data and rebin it
rebinning = 200
_b = ipanema.ristra.get(X)
_c = ipanema.ristra.get(Y)
_prob = mass_model(X, Z, **res.params.valuesdict()) #/ rebinning
assert np.size(_b) % rebinning == 0
_rb = np.sum(_b[i::rebinning] for i in range(rebinning))/rebinning  # averaging
_rc = np.sum(_c[i::rebinning] for i in range(rebinning))  # summing
_rprob = np.sum(_prob[i::rebinning] for i in range(rebinning))  # summing

# _x = ipanema.ristra.linspace(*mass_range, 1000)
# _y = 0 * ipanema.ristra.linspace(*mass_range, 1000)
# _y = mass_model(_x, _y, **res.params.valuesdict())
# _y = mass_model(_x, _y, **pars.valuesdict())
# _x = ipanema.ristra.get(_x)
# _y = ipanema.ristra.get(_y) * hdat.norm
# _y = ipanema.ristra.get(_y) * np.trapz(ipanema.ristra.get(Y), ipanema.ristra.get(X))
# axplot.plot(_x, _y, 'C0')

from scipy.optimize import fsolve
def poisson_Linterval(k, chi2 = 1):
   if k ==0 :
     return [0.,0.5*chi2]
   func = lambda m : m -k - k*np.log(m *1./k) -0.5*chi2
   sols1 = fsolve (func, k-.8*np.sqrt(k))
   sols2 = fsolve (func, k+np.sqrt(k))
   return [k-sols1[0],sols2[0]-k]

errl = []
errh = []
for k in _c:
    l, h = poisson_Linterval(k)
    errl.append(l)
    errh.append(h)
errl = np.array(errl)
errh = np.array(errh)
deltas = _rc - _rprob
_rerrh = np.sum(errh[i::rebinning] for i in range(rebinning)) / np.sqrt(rebinning)  # summing
_rerrl = np.sum(errl[i::rebinning] for i in range(rebinning)) /np.sqrt(rebinning) # summing
_rpulls = np.array([(deltas[i]/_rerrl[i] if deltas[i] > 0 else deltas[i]/_rerrh[i])
                 for i in range(np.size(deltas))])

# axplot.plot(_rb, _rc, '.')
axplot.plot(_rb, _rprob, '-')
# axplot.errorbar(hdat.bins, hdat.counts, yerr=hdat.yerr, fmt='.', color='k')
axplot.errorbar(_rb, _rc, yerr=[_rerrl, _rerrh], fmt='.', color='k')

# _pulls = complot.compute_pdfpulls(_rb, _rprob, _rb, _rc, np.sqrt(_rc), np.sqrt(_rc))
axpull.fill_between(_rb, _rpulls, 0, facecolor="C0", alpha=0.5)
axpull.set_xlabel(r'$m(\pi^+ \pi^- \pi^+)$ [MeV/$c^2$]')
axpull.set_ylim(-6.5, 6.5)
axpull.set_yticks([-5, 0, 5])
axplot.set_ylabel(rf"Candidates")
axplot.set_yscale('log')

plt.show()

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
