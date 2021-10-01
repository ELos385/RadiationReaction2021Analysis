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


X1 = np.copy(X1_original) #+ 0.0005
X2 = np.copy(X2_original) #- 0.0005
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


plotter = False

set1 = [s1_all, s1_all_err, X1]
set2 = [s2_all, s2_all_err, X2]
setw = [swires_all, swires_all_err, Xwires]


#%%

wire_params_12 = []
wire_params_1w = []
wire_params_2w = []

grp1 = set1
grp2 = set2
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
    wire_params_12.append([t_peak, t_width, p_peak, p_width])


grp1 = set1
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
    wire_params_1w.append([t_peak, t_width, p_peak, p_width])
    
 



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
    wire_params_2w.append([t_peak, t_width, p_peak, p_width])

wire_params_12 = np.array(wire_params_12)
wire_params_1w = np.array(wire_params_1w)
wire_params_2w = np.array(wire_params_2w)
#%%

da = (wire_params_12[:,0] - wire_params_1w[:,0])**2 + (wire_params_12[:,2] - wire_params_1w[:,2])**2
da = np.nanmean(da)**(0.5)

db = (wire_params_12[:,0] - wire_params_2w[:,0])**2 + (wire_params_12[:,2] - wire_params_2w[:,2])**2
db = np.nanmean(db)**(0.5)

dc = (wire_params_1w[:,0] - wire_params_2w[:,0])**2 + (wire_params_1w[:,2] - wire_params_2w[:,2])**2
dc = np.nanmean(dc)**(0.5)

print((da,db,dc))