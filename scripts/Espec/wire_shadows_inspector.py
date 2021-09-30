#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
espec wire shadow inspector

Created on Fri Sep 17 23:07:33 2021 by Cary Colgan. 
Email: cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from setup import *

from os import listdir

diag = 'espec1'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/'
onlyfiles = listdir(filepath)
onlyfiles = [i for i in onlyfiles if i[-4:]=='.pkl']

onlyfiles = ['espec1_20210618_run16_Shot020_wire_shadows.pkl']

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


plt.figure()
set_points = np.full((len(onlyfiles), len(epsec2wire_bounds)), fill_value=np.nan)
set_widths = np.full((len(onlyfiles), len(epsec2wire_bounds)), fill_value=np.nan)

for idx, o in enumerate(onlyfiles):
    data = load_object(filepath + o)
    shadows = data['shadows']
    
    x = shadows[:,0]
    y = idx*np.ones_like(shadows[:,0])
    xerr = shadows[:,2]
    
    for x_idx, x0 in enumerate(x):
        for b_idx, (_, lb, ub) in enumerate(epsec1wire_bounds):
            if lb<=x0<=ub:
                set_points[idx, b_idx] = x0
                set_widths[idx, b_idx] = xerr[x_idx]
                plt.axvspan(xmin=lb,xmax=ub, color='r', alpha=0.25, ls='--')
            
    plt.errorbar(x=shadows[:,0], y=idx*np.ones_like(shadows[:,0]), xerr=shadows[:,2], capsize=2, marker='x', ls='', color='k')

plt.xlabel('E axis [mm]')
plt.ylabel('Shot #')
#%%
plt.figure()
offsets = set_points - np.nanmean(set_points, axis=0)
plt.plot(offsets.T, 'x-')

plt.xlabel('Wire #')
plt.ylabel('Offset from avg pixel for this wire')

#%%
plt.figure()

avg_disp = np.nanmean(offsets, axis=1)

offsets -= avg_disp[:, np.newaxis]

plt.plot(offsets, 'x')

plt.show()

# question is whether points tell you MORE than just a constant offset is angle?


