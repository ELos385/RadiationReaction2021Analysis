#GammaSpec_fit_Ec.py
"""
GammaStack Image processing and spectrum fitting for Compton data,
handles one shot at a time.
"""

import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv, kn, expi
from scipy.constants import e, c, epsilon_0, m_e, Boltzmann, hbar, Planck
import math
import matplotlib
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from skimage.io import imread

from lib.general_tools import *
from lib.pipeline import *
from modules.GammaSpec.GammaSpecProc import *


diag='CsIStackSide'#'CsIStackTop'#'CsIStackSide'
# date='20210608'
# run='run10'
# shot='Shot029'


date='20210622'
run='run19'
shot_no_arr=np.array(['004'])#np.array(['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20'])
#
# #load crystal properties from mat file
# coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
# #load energy deposition info from mat file
# Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)
# #load correction factor for CsI stack
# corr_factor_mean, corr_factor_se=load_correction_factor(diag)
# # load image for background subtraction
# path_calib=ROOT_DATA_FOLDER+diag+'/'+date+'/'+bkg_img_dict[date][0]+'/'+"Shot00%s.tif"%(bkg_img_dict[date][1])
# calib_img=imread(path_calib)
#
# CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, hard_hits_filter=None, calib_img=calib_img, kernel=5, debug=False)
CsIStackSide=initialise_gamma_stack_obj('CsIStackSide', date)
CsIStackTop=initialise_gamma_stack_obj('CsIStackTop', date)

if (date=='20210622' and run=='run19'):
    CsIStackTop.norm_factor=1.0
    CsIStackSide.norm_factor=1.0

# parameters for Bayes fitting routine

Bayes_guess=np.array([230.0, 0.05, 0.05])
no_dim=4
no_walkers=(no_dim+1)*2
no_steps=10000
no_burn=8000
err_guess=1000.0
percent_std=np.array([0.1])

guess=CsIStackTop.generate_estimates(Bayes_guess, err_guess, percent_std, no_walkers, no_dim)

Ec_arr=[]
predicted_signal_summed_over_columns_arr=[]
measured_signal_summed_over_columns_arr=[]
color_side=['b', 'r']
color_top=['cyan', 'orange']

for i in range(0, len(shot_no_arr)):
    shot='Shot%s'%shot_no_arr[i]

    pathTop=ROOT_DATA_FOLDER+'CsIStackTop/'+date+'/'+run+'/'+shot+'.tif'
    pathSide=ROOT_DATA_FOLDER+'CsIStackSide/'+date+'/'+run+'/'+shot+'.tif'
    GammaStackTop_Img=imread(pathTop)
    GammaStackSide_Img=imread(pathSide)

    #CsIStack.plot_contours(GammaStack_Img)
    # measured_signal_summed_over_columns_Side=CsIStackSide.get_measured_signal_summed_over_columns(GammaStackSide_Img)#, get_total_counts=False, get_beam_pointing=False)
    measured_signal_summed_over_columns_Top=CsIStackTop.get_measured_signal_summed_over_columns(GammaStackTop_Img.copy())
    measured_signal_summed_over_columns_Side=CsIStackSide.get_measured_signal_summed_over_columns(GammaStackSide_Img.copy())
    #measured_signal_summed_over_columns_arr.append(measured_signal_summed_over_columns)
    #plt.plot(np.arange(len(measured_signal_summed_over_columns_Side)), measured_signal_summed_over_columns_Side, color='r')
    # plt.plot(np.arange(len(measured_signal_summed_over_columns_Top)), measured_signal_summed_over_columns_Top, color='b')
    # plt.show()
    #least squares fitting routine
    # guess=[23.0, 1.0]#, 1.0]#,10000.0, 0.5]
    # popt, pcov=CsIStack.least_sqrs(CsIStack.calc_theoretical_Compton_signal_summed_over_columns, guess, measured_signal_summed_over_columns)
    # popt_normed=popt/CsIStack.norm_factor
    # print('Ec=%s, sigma Ec=%s'%(popt_normed, pcov/CsIStack.norm_factor))
    #Gamma_energy_spec=CsIStack.calc_Compton_energy_spec(popt[0], popt[1])#, popt[2])
    #predicted_signal_summed_over_columns=CsIStack.calc_theoretical_Compton_signal_summed_over_columns(None, popt[0], popt[1])#, popt[2])


    #Bayesian fitting routine
    popt=Bayes_mcmc(guess, no_walkers, no_steps, no_burn, no_dim, measured_signal_summed_over_columns_Top, measured_signal_summed_over_columns_Side, CsIStackTop, CsIStackSide, pool=None)
    print(popt.shape)
    # show corner plot
    # fig = corner.corner(popt)#flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)])
    # plt.show()

    CsIStackSide.plot_bayes_inferred_spec(popt)
    #CsIStackTop.plot_bayes_inferred_spec(popt)
    # plt.plot(CsIStack.Egamma_MeV_interp,dN_dE_test)
    plt.show()

    CsIStackSide.plot_transmission_inferred(popt, measured_signal_summed_over_columns_Side, color_side, diag='CsIStackSide', Invert=False)

    CsIStackTop.plot_transmission_inferred(popt, measured_signal_summed_over_columns_Top, color_top, diag='CsIStackTop', Invert=True)
    plt.show()

    Ec_arr.append(popt[0])
    # predicted_signal_summed_over_columns_arr.append(predicted_signal_summed_over_columns)

# Ec_arr, predicted_signal_summed_over_columns_arr, measured_signal_summed_over_columns_arr=np.asarray(Ec_arr), np.asarray(predicted_signal_summed_over_columns_arr), np.asarray(measured_signal_summed_over_columns_arr)
# mean_Ec, std_Ec=np.mean(Ec_arr), np.std(Ec_arr)
# mean_predicted_signal_summed_over_columns, std_predicted_signal_summed_over_columns=np.mean(predicted_signal_summed_over_columns_arr, axis=0), np.std(predicted_signal_summed_over_columns_arr, axis=0)
# mean_measured_signal_summed_over_columns, std_measured_signal_summed_over_columns=np.mean(measured_signal_summed_over_columns_arr, axis=0), np.std(measured_signal_summed_over_columns_arr, axis=0)

#filter_nos=np.linspace(1, CsIStackSide.N_crystals_X_cuttoff, CsIStackSide.N_crystals_X_cuttoff)

# plt.plot(filter_nos, spec1, color='r', label="Ec =%s"%(Ec1))
# plt.plot(filter_nos, spec2, color='b', label="Ec =%s"%(Ec2))
# plt.plot(filter_nos, spec3, color='g', label="Ec =%s"%(Ec3))
# plt.plot(filter_nos, spec4, color='orange', label="Ec =%s"%(Ec4))
# plt.legend()
# plt.show()


#plt.plot(filter_nos, predicted_signal_summed_over_columns, color='r')
# plt.scatter(filter_nos, mean_measured_signal_summed_over_columns, color='b')
# plt.vlines(filter_nos, mean_measured_signal_summed_over_columns-std_measured_signal_summed_over_columns, mean_measured_signal_summed_over_columns+std_measured_signal_summed_over_columns, color='b', alpha=0.5)
# plt.plot(filter_nos, predicted_signal_summed_over_columns, color='r', label="Ec=%s"%(round(popt_normed[0], 3)))
# plt.fill_between(filter_nos, predicted_signal_summed_over_columns-std_predicted_signal_summed_over_columns, predicted_signal_summed_over_columns+std_predicted_signal_summed_over_columns, color='r', alpha=0.5)
# plt.xlabel('Filter numbers')
# plt.ylabel('Normalised counts')
# plt.title('%s '%diag)
# plt.legend()
# plt.tight_layout()
# #plt.savefig("Brems calib run fitted vs measured signal %s.png"%diag)
# plt.show()

'''
# Brems fitting??? Unfinished
guess=[23.0, 0.197, 1.0]
popt, pcov=CsIStack.least_sqrs(CsIStack.calc_theoretical_Brems_signal_summed_over_columns, guess, measured_signal_summed_over_columns)
print('Ec=%s, sigma Ec=%s'%(popt, pcov))
print(popt[1])

Gamma_energy_spec=CsIStack.calc_Brems_energy_spec(popt[0], popt[1])

# E_axis_test=np.linspace(5.0, 1000, 500)
# spec_test=E_axis_test**-(2.0/3.0)*np.exp(-E_axis_test/guess[0])
# E=CsIStack.Egamma_MeV_interp[0]
# print(E**-(2.0/3.0)*np.exp(-E/guess[0]))

# plt.plot(E_axis_test, spec_test, label='E$_{crit}=%s$'%(guess[0]))
plt.plot(CsIStack.Egamma_MeV_interp, Gamma_energy_spec, label='E$_{crit}=%s$'%(popt[0]))
plt.xlabel('Energy (MeV)')
plt.ylabel('$dN/d\gamma$')
plt.legend()
plt.show()

#predicted_signal_summed_over_columns=CsIStack.calc_theoretical_Brems_signal_summed_over_columns(None, guess[0], guess[1], guess[2])
predicted_signal_summed_over_columns=CsIStack.calc_theoretical_Brems_signal_summed_over_columns(None, popt[0], popt[1], popt[2])
filter_nos=np.linspace(1, CsIStack.N_crystals_X, CsIStack.N_crystals_X)

plt.plot(filter_nos, predicted_signal_summed_over_columns, color='r')
plt.plot(filter_nos, measured_signal_summed_over_columns, color='b')
plt.xlabel('filter numbers')
plt.ylabel('normalised counts')
plt.show()'''

#
# # Bayesian Inference routine
'''
Bayes_guess=np.array([18.5, 0.1787118, 0.8])
no_dim=4
no_walkers=(no_dim+1)*2
no_steps=700
no_burn=600
err_guess=0.01
percent_std=np.array([0.1])

guess=CsIStack.generate_estimates(Bayes_guess, err_guess, percent_std, no_walkers, no_dim)#[Ec_guess, height_guess,]# Ec, height, err_array, percent_std, no_walkers, no_dim
params=CsIStack.Bayes_mcmc(guess, no_walkers, no_steps, no_burn, no_dim, measured_signal_summed_over_columns)

# show corner plot
# fig = corner.corner(params)#flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)])
# plt.show()

CsIStack.plot_bayes_inferred_spec(params)
# plt.plot(CsIStack.Egamma_MeV_interp,dN_dE_test)
plt.show()

CsIStack.plot_transmission_inferred(params, measured_signal_summed_over_columns)
plt.show()

Bayesian Optimisation routine
spec_b_opt, err_spec_b_opt=CsIStack.bayesian_opt_spec(measured_signal_summed_over_columns)
spec_b_opt=CsIStack.bayesian_opt_spec(measured_signal_summed_over_columns)
plt.plot(CsIStack.Egamma_MeV_interp, spec_b_opt, color='r')
plt.show()'''



# x_sampled=np.uniform(np.min(GammaStack.Egamma_MeV_interp), np.max(GammaStack.Egamma_MeV_interp), 1)
# A=0.15
# Ec=40.0
# E=np.linspace(0.0, 80.0, 200)
# dN_dE_inv_compt=A*(E/Ec)**(-2.0/3.0)*np.exp(E/Ec)

# A_arr=np.array([0.15, 0.6, 0.34])
# Ec_arr=np.array([35.0, 23.0, 46.5])

'''
def dN_dE_inv_compt(E, Ec, A):
    return A*(E/Ec)**(-2.0/3.0)*np.exp(E/Ec)

def calc_Brems_energy_spec(E, Te_MeV, B):
    gff=3.0**0.5/np.pi*np.exp(E/(2.0*Te_MeV))*kn(0, E/(2.0*Te_MeV))#expi(0.5*(E/Te_MeV)**2)
    dP_dE_brehms=B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)*gff
    # print("gff=%s"%gff)
    # print("B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)=%s"%(B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)))
    return dP_dE_brehms

def gaussian_spec(E, mean, sigma, height):
    return height*np.exp(-(E-mean)**2/(2.0*sigma**2))
'''
#train 2000 on each spectrum type, 4000 on summations of the three
#create test data, 1000 points, which contains all three
#
# summed_I_compt=0
# for i in range(0, len(A_arr)):
#     summed_I_compt+=dN_dE_inv_compt(E, Ec_arr[i], A_arr[i])
# plt.plot(E, summed_I_compt, color='r')
# plt.plot(E, dN_dE_inv_compt(E, Ec_arr[2], A_arr[2]), color='b')
# plt.show()


#
#
# # power spec for brems from a plasma (assuming electrons folloe a maxwell boltzman distribution)
# wL=c*2.0*np.pi/(800.0*10**-9)
# Zhe=2
# ne_per_cm_3=1.0*10.0**18# N electrons per cm**3
# Te_MeV=20
# # Te_J=Te_MeV/e*10**6
#
#
# Te_eV=10**9#Te_K*Boltzmann/e
# Te_K=Te_eV*e/Boltzmann
# w_wL=np.linspace(0.01, 0.5*10**8, 1000)# photon frequency
#
# y=0.5*(w_wL*wL*hbar/(Te_K*Boltzmann))**2
#
#
# print('Te_eV =%s'%Te_eV)
#
# ne_per_m_3=ne_per_cm_3*10**6
# nhe_per_m_3=ne_per_m_3/2.0
# wp_rad_per_s=(ne_per_m_3*e**2/(epsilon_0*m_e))**0.5# rad per second
# wp_wL=wp_rad_per_s/wL
# Ep=wp_rad_per_s*hbar/(e*10**6)
# #e bunch in wakefield has GeV temp
# # Te_K=298*()
# # print(y)
# # print((wp_wL/w_wL))
# # print(expi(y))
# # print(1.0/(m_e*c**2)**1.5)
# # print((e**2/(4.0*np.pi*epsilon_0))**3)
# # print(ne_per_m_3*nhe_per_m_3)
# # print(Te_K*Boltzmann)
# print(8.0*2.0**0.5/(3.0*np.pi**0.5)*(e**2/(4.0*np.pi*epsilon_0))**3*1.0/(m_e*c**2)**1.5*Zhe*nhe_per_m_3*ne_per_m_3/(Te_K*Boltzmann)**0.5)
# #brehms from single particle
# #dN_dE_brehms=8.0*2.0**0.5/(3.0*np.pi**0.5)*(e**2/(4.0*np.pi*epsilon_0))**3*1.0/(m_e*c**2)**1.5*(1.0-wp_wL**2/w_wL**2)**0.5*Zhe*nhe_per_m_3*ne_per_m_3/(Te_K*Boltzmann)**0.5*expi(y)
# # brehms from plasma
# # print("gff = %s"%gff)
# print(8.0/3.0*(2.0*np.pi/3.0)**0.5)
# print(e**6/(m_e**1.5*c**3))
# print(m_e**1.5*c**3)
# print(e**6/(m_e**1.5*c**3)*nhe_per_m_3*5.0*ne_per_m_3)
# print(1.0/(Te_K*Boltzmann)**0.5)
# print(np.exp(-w_wL*wL*hbar/(Te_K*Boltzmann)))
# print(8.0/3.0*(2.0*np.pi/3.0)**0.5*e**6/(m_e**0.5*c)**3*nhe_per_m_3*ne_per_m_3/(Te_K*Boltzmann)**0.5)
# # gff=3.0**0.5/np.pi*np.log(Te_eV*10**-6/(E))
# #dN_dE_brehms=8.0/3.0*(2.0*np.pi/3.0)**0.5*e**6/(m_e**0.5*c*hbar)**3*nhe_per_m_3*ne_per_m_3/(Te_K*Boltzmann)**0.5*np.exp(-w_wL*wL*hbar/(Te_K*Boltzmann))*gff
# # B had units of MeV**0.5
# # B=10.0
# gff=3.0**0.5/np.pi*np.log(Te_eV*10**-6/(E))
# dN_dE_brehms=B*1.0/(Te_eV*10**-6)**0.5*np.exp(-E/(Te_eV*10**-6))*gff
#
# ####CHECK THIS!!!!!!
# print(wL)
# print(hbar*wL)
# print(wL*hbar*e)
# plt.plot(E, dN_dE_brehms, color='r')
# plt.plot(E, dN_dE_inv_compt, color='b')
# plt.show()
# filter_nos=np.linspace(1, CsIStack.N_crystals_X, CsIStack.N_crystals_X)
#
