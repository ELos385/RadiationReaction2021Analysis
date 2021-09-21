#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
espec wire shadow inspector

Created on Fri Sep 17 23:07:33 2021 by Cary Colgan. 
Email: cary.colgan13@imperial.ac.uk
"""
import numpy as np
import matplotlib.pyplot as plt

from os import listdir

diag = 'espec1'
filepath = HOME + '/results/Espec/wire_shadows/' + diag + '/'
onlyfiles = listdir(filepath)
onlyfiles = [i for i in onlyfiles if i[-4:]=='.pkl']

bounds = [[85, 95],
          [95, 105],
          [110,115],
          [118, 123],
          [123, 126],
          [135, 140],
          [147, 152],
          [160, 165],
          [165, 170], 
          [173, 178],
          [185, 190],
          [198, 203],
          [210, 215],
          [218, 223],
          [223, 228]
        ]

plt.figure()
set_points = np.full((len(onlyfiles), len(bounds)), fill_value=np.nan)
set_widths = np.full((len(onlyfiles), len(bounds)), fill_value=np.nan)

for idx, o in enumerate(onlyfiles):
    data = load_object(filepath + o)
    shadows = data['shadows']
    
    x = shadows[:,0]
    y = idx*np.ones_like(shadows[:,0])
    xerr = shadows[:,2]
    
    for x_idx, x0 in enumerate(x):
        for b_idx, (lb, ub) in enumerate(bounds):
            if lb<=x0<=ub:
                set_points[idx, b_idx] = x0
                set_widths[idx, b_idx] = xerr[x_idx]
            
    plt.errorbar(x=shadows[:,0], y=idx*np.ones_like(shadows[:,0]), xerr=shadows[:,2], capsize=2, marker='x', ls='', color='k')

plt.xlabel('E axis [pixels]')
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


