#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Wed Sep 29 19:40:59 2021 by Cary Colgan. 
Email: cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *

from modules.Espec.espec_wire_tracking import *



#%%
# grab calibration data
filedir = HOME + '/calib/'
file_ext = '_2screen_disp.pkl'
espec1_spec = load_object(filedir + 'espec1/espec1' + file_ext)
espec2_spec = load_object(filedir + 'espec2/espec2' + file_ext)
especwires_spec = load_object(filedir + 'especwires/especwires' + file_ext)

E_MeV = espec1_spec['X_e_t_E_MeV']
t_mrad = espec1_spec['t_mrad']
T_axis_original, P_axis_original = np.meshgrid(E_MeV, t_mrad)
X1_original = np.copy(espec1_spec['X_e_t'])

X2_original = np.copy(espec2_spec['X_e_t'])

Xwires_original = np.copy(especwires_spec['X_e_t'])

#plt.figure()
    #cax = plt.pcolormesh(T_axis, P_axis, X1_original)
#plt.colorbar(cax)


#%%
# Issue of large sample space and very small measurements here.
# Strategy is to artificially increase uncertainty in shadow measurement, then
# home in on the ROI, and decrease size of param space to allow resolution of 
# param space to increase
#
# seems to work quickly, with only 4 iterations needed.

# wire_shadow_idx (0-19), lb [mm], ub [mm]
#epsec1wire_bounds = np.array(np.copy(epsec1wire_bounds))*[1, 1e-3, 1e-3]
#epsec2wire_bounds = np.array(np.copy(epsec2wire_bounds))*[1, 1e-3, 1e-3]    

# iterate solving over smaller and smaller regions, so doesn't require too much effort
ex = '20210618_run02_Shot031_wire_shadows.pkl'
#ex = '20210618_run16_Shot020_wire_shadows.pkl'

diag = 'espec1'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/' + diag + '_'
data = load_object(filepath + ex)
shadows1 = data['shadows']
# shadows format is:
# shadow_mid_point, shadow_mid_point_err, shadow_width (std), shadow_width_err
s1_all = shadows1[:,0]*1e-3
s1_all_err = shadows1[:,1]*1e-3
s1_all, s1_all_err = get_full_shadows(s1_all, s1_all_err, diag)


diag = 'espec2'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/' + diag + '_'
data = load_object(filepath + ex)
shadows2 = data['shadows']
s2_all = shadows2[:,0]*1e-3
s2_all_err = shadows2[:,1]*1e-3
s2_all, s2_all_err = get_full_shadows(s2_all, s2_all_err, diag)

swires_all = especwires_spec['wire_dist']
frac = 0.1
swires_all_err = np.array([450e-6 * frac]*10 + [380e-6 * frac]*10)


X1 = np.copy(X1_original)
X2 = np.copy(X2_original)
Xwires = np.copy(Xwires_original)  - 0.0015
p_axis = np.copy(P_axis_original[:,0])
t_axis = np.copy(T_axis_original[0,:])


# conv to get increasing in all axes for interpolation
X1 = np.flipud(X1)
X2 = np.flipud(X2)
Xwires = np.flipud(Xwires)
p_axis = p_axis[::-1]

# for testing = quick
#ds_decrease_factor_on_iter = 2 # takes 3 iterations to get it exact
#refine_factor_on_iter = 5

wire_params = []
plotter = False

hole_idx = 18
#for s1,s1_err,s2,s2_err in list(zip(swires_all, swires_all_err, s2_all, s2_all_err))[hole_idx:hole_idx+1]:
#for s1,s1_err,s2,s2_err in list(zip(s1_all, s1_all_err, s2_all, s2_all_err))[hole_idx:hole_idx+1]:

set1 = [s1_all, s1_all_err, X1]
set2 = [s2_all, s2_all_err, X2]
setw = [swires_all, swires_all_err, Xwires]


# quick way to change what two screens are being compared
grp1 = set2
grp2 = setw
for s1, s1_err, s2, s2_err in list(zip(grp1[0],grp1[1], grp2[0], grp2[1])):
    X1 = grp1[2]
    X2 = grp2[2]
    if np.any(np.isnan([s1,s2])):
        t_peak, t_width, p_peak, p_width = [np.nan]*4
    else:
        t_peak, t_width, p_peak, p_width = find_wire_shadow_param_space(X1, X2, t_axis, p_axis, 
                                     s1, s1_err, s2, s2_err,
                                     number_of_iterations=4,
                                     ds_factor_start=1000, ds_decrease_factor_on_iter=10, refine_factor_on_iter=10,
                                     lim=1e-2, plotter=plotter)
    wire_params.append([t_peak, t_width, p_peak, p_width])
    
wire_params=np.array(wire_params)
#%%
def J(x, pcov):
    """only works if pcov came from a polynomial fit, so dy/dp obvious
    """
    n_params = pcov.shape[0]
    dp = np.abs(np.array([x**(i) for i in list(range(n_params))[::-1]])).T

    #a = np.dot(pcov, dp, axis=-1)
    #err2 = np.dot(dp.T, a)

    # i,j are 2x2 matrix etc. axes for pcov
    # k is number of x points
    a = np.einsum('ij,kj->ki',pcov,dp)
    err2 = np.einsum('ki,ki->k',dp,a)

    return err2**(0.5)

plt.figure()
x = wire_params[:,0]
xerr=  wire_params[:,1]
y = wire_params[:,2]
yerr =  wire_params[:,3]

x = x[np.isfinite(yerr)]
xerr = xerr[np.isfinite(yerr)]
y = y[np.isfinite(yerr)]
yerr = yerr[np.isfinite(yerr)]

plt.errorbar(x,y,xerr,yerr,marker='x', ls='', capsize=2, color='k')

plt.xlabel('$E$ [MeV]')
plt.ylabel('$\\theta(E)$ [mrad]')

popt, pcov = np.polyfit(x,y,deg=3,cov=True)
perr = np.diagonal(pcov)**(0.5)
z = np.poly1d(popt)
dx = np.linspace(x.min(), x.max(), 100)
plt.plot(dx, z(dx), 'r')

plt.fill_between(dx, z(dx)-J(dx, pcov), z(dx)+J(dx,pcov), color='r', alpha=0.25)

#'20210618_run20_Shot048_wire_shadows.pkl'
ex_name = ex.split('_')[0:3]
d,r,s = ex_name
plt.title('%s %s %s' % (d,r,s))

#%%
"""
# example fig of what param space marked out by each shadow measurement
#
# have to artifically increase widths to see anything
# upsample so you can actually see something as well

plt.figure()

N = 1000

tl, tr = 400.0, 1000.0
pl, pr = -5.0, +30.0


# repeat but upsampled, in the vicinity of tl-tr, pl-pr
new_t_axis = np.linspace(tl, tr, N)
new_p_axis = np.linspace(pl, pr, N)
new_P_axis, new_T_axis = np.meshgrid(new_p_axis, new_t_axis)

X1 = np.flipud(X1_original)
X2 = np.flipud(X2_original)
Xwires = np.flipud(Xwires_original)
P_axis = np.flipud(P_axis_original)

fZ1 = RectBivariateSpline(P_axis[:,0], T_axis_original[0,:], X1) 
X1_upsampled = fZ1(new_p_axis, new_t_axis)

fZ2 = RectBivariateSpline(P_axis[:,0], T_axis_original[0,:], X2)
X2_upsampled = fZ2(new_p_axis, new_t_axis)

fZw = RectBivariateSpline(P_axis[:,0], T_axis_original[0,:], Xwires)
Xw_upsampled = fZw(new_p_axis, new_t_axis)

X1c = np.zeros_like(X1_upsampled)
X2c = np.zeros_like(X2_upsampled)
Xwc = np.zeros_like(Xw_upsampled)

for s, s_err in zip(s1_all, 100.0*s1_all_err):
    if np.isnan(s):
        pass
    else:
        X_s = normal(X1_upsampled, s, s_err)
        X1c += X_s
        
for s, s_err in zip(s2_all, 100.0*s2_all_err):
    if np.isnan(s):
        pass
    else:
        X_s = normal(X2_upsampled, s, s_err)
        X2c += X_s

for s, s_err in zip(swires_all, 10.0*swires_all_err):
    if np.isnan(s):
        pass
    else:
        X_s = normal(Xw_upsampled, s, s_err)
        Xwc += X_s

X1c[X1c<0.1*np.nanmax(X1c)] = np.nan
X2c[X2c<0.1*np.nanmax(X2c)] = np.nan
Xwc[Xwc<0.1*np.nanmax(Xwc)] = np.nan

cax1 = plt.pcolormesh(new_T_axis, new_P_axis, X1c.T, cmap='Reds')
cax2 = plt.pcolormesh(new_T_axis, new_P_axis, X2c.T, cmap='Blues')
cax3 = plt.pcolormesh(new_T_axis, new_P_axis, Xwc.T, cmap='Greens')

#plt.figure()
#cax2 = plt.pcolormesh(new_T_axis, new_P_axis, X1c.T * X2c.T)
#plt.errorbar(x,y,xerr,yerr,marker='x', ls='', capsize=2, color='k')

#plt.plot(x,y,marker='.',ls='',color='k')
#plt.colorbar(cax1)
#plt.colorbar(cax2)

plt.xlabel('$E$ [MeV]')
plt.ylabel('$\\theta(E)$ [mrad]')
plt.title('Screen1 red, Screen2 blue, Wires green')
"""