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

def normal(X, mu, fwhm):
    sigma = fwhm / 2.355
    pre = sigma * (2.0*np.pi)**(0.5)
    index = (X-mu)**2 / (2.0 * sigma**2)
    return pre * np.exp(-index)



def normalise(z, x, y=None):
    """Normalise a 2D or 1D distribution
    """
    s = np.trapz(z, x)
    if y is not None:
        s = np.trapz(s, y)
    s = np.abs(s)
    return z.copy()/s

def find_FWHM(x,y,frac=0.5):
    """Brute force FWHM calculation.
    Frac allows you to easily change, so to e-2 value etc.
    
    """
    fm = y.copy() - y.min()
    fm = fm.max()
    hm = fm * frac

    hm += y.min()
    fm = fm + y.min()
    max_idx = np.argmax(y)
    
    first_half = np.arange( int(0.9 * max_idx) )
    second_half = np.arange( int(1.1 * max_idx), x.size )
    
    hm_x_left = np.interp(hm, y[first_half], x[first_half])
    hm_x_right = np.interp(hm, y[second_half][::-1], x[second_half][::-1])
    
    return hm_x_right - hm_x_left

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
    
from scipy.interpolate import RectBivariateSpline

def find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=1e-5,
                             plotter=True):
    """Given electron tracking maps, function returns the parameter space 
    (E_min, E_max, theta_min, theta_max) range that the inner product of the 
    shadow measurements, s1 ± ds1, s2 ± ds2, cover.
    
    Range is defined as the everything >= (lim * peak value) of the 
    normalised inner product.
    """
    # get overlap region
    X1, X2 = np.copy(X1), np.copy(X2)
    t_axis, p_axis = np.copy(t_axis), np.copy(p_axis)
    
    T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    X1 *= normal(X1, s1, ds1)
    X2 *= normal(X2, s2, ds2)
    X_prod = X1 * X2
    X_prod = normalise(X_prod, t_axis, p_axis)
    X_sum = X1 + X2
    
    if plotter == True:
        plt.figure()
        cax = plt.pcolormesh(T_axis, P_axis, X_prod)
        plt.colorbar(cax)
        plt.title('$X_{1} \cdot X_{2}$')
        
        plt.figure()
        cax = plt.pcolormesh(T_axis, P_axis, X_sum)
        plt.colorbar(cax)
        plt.title('$X_{1} + X_{2}$')
    
    # find parameter space the overlap occupies 
    # roi in parameter space is min and max where 
    # X_prod >= X_prod * lim 
    lineout1 = np.nanmean(X_prod, axis=0)
    lineout1_lim = np.nanmax(lineout1)*lim
    t_valid = t_axis[lineout1 >= lineout1_lim]
    if t_valid.size == 0:
        print('Disjoint PDFs')
        tl, tr= np.nan, np.nan
    else:
        tl, tr = t_valid[0], t_valid[-1]
    t_peak = t_axis[np.argmax(lineout1)]
    
    lineout2 = np.nanmean(X_prod, axis=1)
    lineout2_lim = np.nanmax(lineout2)*lim
    p_valid = p_axis[lineout2 >= lineout2_lim]
    if p_valid.size == 0:
        print('Disjoint PDFs')
        pl, pr = np.nan, np.nan
    else:
        pl, pr = p_valid[0], p_valid[-1]
    p_peak = p_axis[np.argmax(lineout2)]

    if plotter == True:
        plt.figure()    
        plt.plot(t_axis, lineout1, color=colors[0])
        plt.axhline(y=lineout1_lim, color=colors[0])    
        plt.axvline(x=tl, color=colors[0], ls='--')
        plt.axvline(x=tr, color=colors[0], ls='--')
        
        plt.plot(p_axis, lineout2, color=colors[1])
        plt.axhline(y=lineout2_lim, color=colors[1])
        plt.axvline(x=pl, color=colors[1], ls='--')
        plt.axvline(x=pr, color=colors[1], ls='--')
        plt.title('sub-region')
    
    if tr<tl: tl,tr=tr,tl
    if pr<pl: pl,pr=pr,pl
    
    return t_peak, p_peak, tl, tr, pl, pr


def find_wire_shadow_param_space(X1, X2, t_axis, p_axis, 
                                 s1, s1_err, s2, s2_err,
                                 number_of_iterations=4,
                                 ds_factor_start=1000, ds_decrease_factor_on_iter=10, refine_factor_on_iter=10,
                                 lim=1e-2, plotter=True):
    """iterates find_overlap_param_space a number of times. On each iteration 
    the parameter space is restricted to lim * peak, and the artificial 
    enhancing of the error on s1, s2 is decreased by ds_decrease_factor_on_iter
    and the resolution in the sample area is increased by refine_factor_on_iter.
    
    Returns t_peak, t_width, p_peak, p_width
    
    """
    T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    for i in range(1, number_of_iterations+1):
        ds_factor = ds_factor_start / (ds_decrease_factor_on_iter)**(i-1) # start high
        ds1 = s1_err * ds_factor
        ds2 = s2_err * ds_factor
        #print('Iteration %s. ds_factor = %s' % (i,ds_factor))
        
        t_peak, p_peak, tl, tr, pl, pr = find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=lim, plotter=plotter)
        
        check = np.array([t_peak, p_peak, tl, tr, pl, pr])
        if np.any(np.isnan(check)):
            return tuple([np.nan]*4)
        
        upsample_factor = refine_factor_on_iter
        
        # repeat but upsampled, in the vicinity of tl-tr, pl-pr
        new_t_axis = t_axis[ (t_axis>=tl) & (t_axis<=tr) ]
        new_t_axis = np.linspace(new_t_axis.min(), new_t_axis.max(), upsample_factor * new_t_axis.size)
        new_p_axis = p_axis[ (p_axis>=pl) & (p_axis<=pr) ]
        new_p_axis = np.linspace(new_p_axis.min(), new_p_axis.max(), upsample_factor * new_p_axis.size)
        new_P_axis, new_T_axis = np.meshgrid(new_p_axis, new_t_axis)
        
        # null - just refines what's already there
        #new_t_axis = np.linspace(t_axis.min(), t_axis.max(), upsample_factor*t_axis.size)
        #new_p_axis = np.linspace(p_axis.min(), p_axis.max(), upsample_factor*p_axis.size)
        
        fZ1 = RectBivariateSpline(P_axis[:,0], T_axis[0,:], X1) 
        X1_upsampled = fZ1(new_p_axis, new_t_axis)
        
        fZ2 = RectBivariateSpline(P_axis[:,0], T_axis[0,:], X2)
        X2_upsampled = fZ2(new_p_axis, new_t_axis)
    
        # to pass onto next iteration
        X1, X2, t_axis, p_axis = X1_upsampled, X2_upsampled, new_t_axis, new_p_axis
        T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    # do final time
    ds1 = s1_err 
    ds2 = s2_err
    t_peak, p_peak, tl, tr, pl, pr = find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=0.5, plotter=plotter)
    
    t_width = 0.5 * (tr - tl)
    p_width = 0.5 * (pr - pl)
    
    return t_peak, t_width, p_peak, p_width

# wire_shadow_idx (0-19), lb [mm], ub [mm]
epsec1wire_bounds = [[5, 85, 95],
          [6, 95, 105],
          [7, 110,115],
          [8, 118, 123],
          [9, 123, 126],
          [10, 135, 140],
          [11, 147, 152],
          [12, 160, 165],
          [13, 165, 170], 
          [14, 173, 178],
          [15, 185, 190],
          [16, 198, 203],
          [17, 210, 215],
          [18, 218, 223],
          [19, 223, 228]
        ]

epsec2wire_bounds = [[5, 60, 70],
          [6, 80, 90],
          [7, 97,106],
          [8, 107, 115],
          [9, 117, 125],
          [10, 135, 145],
          [11, 155, 165],
          [12, 170, 180],
          [13, 182, 190], 
          [14, 190, 200],
          [15, 210, 217],
          [16, 228, 238],
          [17, 247, 255],
          [18, 256, 265],
          [19, 266, 275]
        ]

epsec1wire_bounds = np.array(epsec1wire_bounds)*[1, 1e-3, 1e-3]
epsec2wire_bounds = np.array(epsec2wire_bounds)*[1, 1e-3, 1e-3]


def get_full_shadows(s,s_err, bounds_array):
    """Given list of shadows and errors, function returns them in an array
    of size = 20, for all the wires used on experiment, with their values in the
    right position.
    
    list idx 0 is the high energy end of the shadows, idx -1 is the low energy end.
    
    bounds array should be np.array of shape(N,3) for N wires, where 
    col 0 is shadow id (0 being highest energy, 19 being lowest)
    col 1 is lower bound of where it could be on screen [m]
    col 2 is upper bound [m]
    """
    new_shadows = np.full((20), fill_value=np.nan)
    new_shadows_err = np.full((20), fill_value=np.nan)
    for i,j in zip(s, s_err):
        ids = (i >= bounds_array[:,1]) & (i <= bounds_array[:,2])
        idx = int(bounds_array[ids,0][0])
        new_shadows[idx] = i
        new_shadows_err[idx] = j
    return new_shadows, new_shadows_err

    


# iterate solving over smaller and smaller regions, so doesn't require too much effort
ex = '20210618_run20_Shot048_wire_shadows.pkl'
#ex = '20210618_run16_Shot020_wire_shadows.pkl'

diag = 'espec1'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/' + diag + '_'
data = load_object(filepath + ex)
shadows1 = data['shadows']
# shadows format is:
# shadow_mid_point, shadow_mid_point_err, shadow_width (std), shadow_width_err
s1_all = shadows1[:,0]*1e-3
s1_all_err = shadows1[:,1]*1e-3
s1_all, s1_all_err = get_full_shadows(s1_all, s1_all_err, epsec1wire_bounds)


diag = 'espec2'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/' + diag + '_'
data = load_object(filepath + ex)
shadows2 = data['shadows']
s2_all = shadows2[:,0]*1e-3
s2_all_err = shadows2[:,1]*1e-3
s2_all, s2_all_err = get_full_shadows(s2_all, s2_all_err, epsec2wire_bounds)

swires_all = especwires_spec['wire_dist']
frac = 0.1
swires_all_err = np.array([450e-6 * frac]*10 + [380e-6 * frac]*10)


X1 = np.copy(X1_original) #+ 0.002
X2 = np.copy(X2_original) #- 0.003
Xwires = np.copy(Xwires_original) 
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
grp1 = set1
grp2 = set2
for s1, s1_err, s2, s2_err in list(zip(grp1[0],grp1[1], grp2[0], grp2[1])):
    X1 = grp1[2]
    X2 = grp2[2]
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


#%%
# example fig - upsample so you can actually see something!
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