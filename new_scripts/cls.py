import numpy as np
import math
from scipy.interpolate import interp1d

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from classy import Class
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.special import hyp2f1
import sys,os
from time import time
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special
from scipy.special import factorial

#Starting CLASS

z_pk = 0
common_settings = {# fixed LambdaCDM parameters
                   'A_s':2.089e-9,
                   'n_s':0.9649,
                   'tau_reio':0.052,
                   'omega_b':0.02237,
                   'h':0.6736,
                   'YHe':0.2425,
#                   'N_eff':3.046,
                   'N_ur':2.0328,
                   'N_ncdm':1,
#                   'N_ncdm':0,
                   'm_ncdm':0.06,
                   # other output and precision parameters
#                    'P_k_max_1/Mpc':100.0,
                   'z_pk':z_pk,
                   'output':'mPk,tCl,pCl,lCl',
                   'lensing':'yes',
                   'l_max_scalars':9000.}

M = Class()
M.set(common_settings)
#compute linear LCDM only
M.set({ 'non linear':'no',
        'omega_cdm': 0.1193,
      })

M.compute()

#compute n=0
M1 = Class()
M1.set(common_settings)
M1.set({'non linear':'no',
        'omega_cdm': 1e-15,
        'omega_dmeff': 0.1193,
        'sigma_dmeff': 5e-27,
        'npow_dmeff':0,
        'm_dmeff':1e-3,
        'Vrel_dmeff':0,
        'dmeff_target':'baryons'
       })
M1.compute()

#compute n=2
M2 = Class()
M2.set(common_settings)
M2.set({'non linear':'no',
        'omega_cdm': 1e-15,
        'omega_dmeff': 0.1193,
        'sigma_dmeff': 7.5e-22,
        'npow_dmeff':2,
        'm_dmeff':1e-3,
        'Vrel_dmeff':0,
        'dmeff_target':'baryons'
       })
M2.compute()

#Extracting and plotting spectra

bg_LCDM = M.get_background()
th_LCDM = M.get_thermodynamics()
cl_LCDM = M.lensed_cl()

bg_0 = M1.get_background()
th_0 = M1.get_thermodynamics()
cl_0 = M1.lensed_cl()

bg_2 = M2.get_background()
th_2 = M2.get_thermodynamics()
cl_2 = M2.lensed_cl()

h = M.h()
k = np.linspace(log(0.0001),log(50),200)
k = np.exp(k)
twopi = 2.*math.pi
khvec = k*h

fig_tt, ax_tt = plt.subplots()
fig_ee, ax_ee = plt.subplots()

ax_tt.set_xlabel('$\ell$', fontsize=21, labelpad=12)
ax_tt.set_ylabel(r'$C_\ell^{\mathrm{TT}}/C_\ell^{\mathrm{TT , ref}}$', fontsize=21, labelpad=12)
ax_tt.set_xlim([1,5000])

ax_ee.set_xlabel('$\ell$', fontsize=21, labelpad=12)
ax_ee.set_ylabel(r'$C_\ell^{\mathrm{EE}}/C_\ell^{\mathrm{EE , ref}}$', fontsize=21, labelpad=12)
ax_ee.set_xlim([1,5000])

l = np.array(cl_LCDM['ell'])
l = np.delete(l, 0)
l = np.delete(l, 0)

tt_LCDM = np.array(cl_LCDM['tt'])
tt_LCDM = np.delete(tt_LCDM, 0)
tt_LCDM = np.delete(tt_LCDM, 0)

ee_LCDM = np.array(cl_LCDM['ee'])
ee_LCDM = np.delete(ee_LCDM, 0)
ee_LCDM = np.delete(ee_LCDM, 0)

tt_0 = np.array(cl_0['tt'])
tt_0 = np.delete(tt_0, 0)
tt_0 = np.delete(tt_0, 0)

ee_0 = np.array(cl_0['ee'])
ee_0 = np.delete(ee_0, 0)
ee_0 = np.delete(ee_0, 0)

tt_2 = np.array(cl_2['tt'])
tt_2 = np.delete(tt_2, 0)
tt_2 = np.delete(tt_2, 0)

ee_2 = np.array(cl_2['ee'])
ee_2 = np.delete(ee_2, 0)
ee_2 = np.delete(ee_2, 0)

#Plotting spectra

ax_tt.plot(l, tt_0/tt_LCDM, color = 'purple', label = 'n=0')
ax_tt.plot(l, tt_2/tt_LCDM, color = 'orange', label = 'n=2')

tt_errors = np.genfromtxt('Dl_all_TT_cmb_only.dat')
l_errors = tt_errors[:,0]
minus_two_sigma = 2*tt_errors[:,2]/tt_errors[:,1]
plus_two_sigma = 2*tt_errors[:,2]/tt_errors[:,1]

ax_tt.fill_between(l_errors,1+plus_two_sigma,1-minus_two_sigma,alpha=0.5,color='lightgray')
ax_tt.set_ylim(0.6,1.4)
ax_tt.legend(fontsize='14',ncol=1,loc='upper right')

fig_tt.tight_layout()
fig_tt.savefig('Cl_tt_n0_n2.pdf')

ax_ee.plot(l, ee_0/ee_LCDM, color = 'purple', label = 'n=0')
ax_ee.plot(l, ee_2/ee_LCDM, color = 'orange', label = 'n=2')

ee_errors = np.genfromtxt('/home1/adamhe/planck_data/cl_error_bars/COM_PowerSpect_CMB-EE-binned_R3.02.txt')
l_errors = ee_errors[:,0]
minus_two_sigma = 2*ee_errors[:,2]/ee_errors[:,1]
plus_two_sigma = 2*ee_errors[:,3]/ee_errors[:,1]

ax_ee.fill_between(l_errors,1+plus_two_sigma,1-minus_two_sigma,alpha=0.5,color='lightgray')
ax_ee.set_ylim(0,2)
ax_ee.legend(fontsize='14',ncol=1,loc='upper right')

fig_ee.tight_layout()
fig_ee.savefig('Cl_ee_n0_n2.pdf')

#plt.show()

#Cleaning up

M.struct_cleanup()
M.empty()
M1.struct_cleanup()
M1.empty()
M2.struct_cleanup()
M2.empty()
