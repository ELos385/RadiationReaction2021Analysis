#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Thu Sep 16 11:43:03 2021 by Cary Colgan. 
Email: cary.colgan13@imperial.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *

from modules.Espec.espec_wire_tracking import find_wire_shadows

# start with an example shot - same workflow as used in LivePlotting

#date = '20210618'
date = '20210621'
run = 'run18'

espec_diag_names = ['espec1', 'espec2']
file_ext = '.tif'


# make objects for holding espec analysis data
specs = []
for n, espec_diag_name in enumerate(espec_diag_names):
    run_name = date+'/run99'

    spec = Espec(run_name,shot_num=1,img_bkg=None,diag=espec_diag_name)
    specs.append(spec)

#%%
shot='Shot020'



# grab a shot's images - like LivePlotting would do 
fig, axes = plt.subplots(2,1)
#fig2, axes2 = plt.subplots(2,1, sharex=True)

screen_distances = [1740.0, 2300.0]

data = []
for idx, spec in enumerate(specs): 
    filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot + file_ext
    im = spec.get_image(filepath)
    x_mm, y_mm = spec.x_mm, spec.y_mm
    
    left,right,bottom,top = x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()
    extent = left,right,bottom,top
    axes[idx].imshow(im, extent=extent)
    if hasattr(spec, 'p_labels'):
        for p in np.array(spec.xaxis)[:,0]:
            pass
            #win.docks[spec].widgets[0].view.addLine(x=p) 
            #axes[idx].axvline(x=p, color='w', alpha=0.5)
    

    # try plotting w.r.t energy axis instead
    x, y = spec.x_MeV, spec.y_mm
    y = y/screen_distances[idx]
    
    y -= np.nanmean(y)
    
    cols = ~np.isnan(x)
    #axes2[idx].pcolormesh(x[cols], y, im[:, cols])
    dicto = {'x_MeV': x,
             'x_mm': x_mm,
             'y': y,
             'im': im}
    data.append(dicto)

#axes[1].set_xlabel('x [pixels]')

#axes2[0].set_xlim((1500, 350))
#print('finished')

analysis_kwargs = {'n_sigma': 1.0,
          'kernel_size': 5,
          
          'prominence': 2e-2,
          'distance': 40,
          'wire_width': 5,
          
          'smoothing_factor': 0.1,
        }

fig, ax = plt.subplots()
sizes = []
for idx, dicto in enumerate(data):
    #x,y,im = dicto['x_MeV'], dicto['y'], dicto['im']
    x_mm, y, im, x_MeV = np.copy(dicto['x_mm']), np.copy(dicto['y']), np.copy(dicto['im']), np.copy(dicto['x_MeV'])
    
    shadows = find_wire_shadows(x_mm, im, plotting=True, **analysis_kwargs)
    
        
    E_vals = np.interp(shadows[:,0], x_mm, x_MeV)
    
    ax.errorbar(x=shadows[:,0], y=E_vals, xerr=shadows[:,2], linestyle='', marker='x', capsize=2, label='espec%g' % (idx+1))
    sizes.append(shadows[:,0].size)

ax.set_xlabel('x [mm]')
ax.set_ylabel('T($\\theta=0$) [MeV]')
ax.grid()
ax.legend()

print('Same number on both screens?: ', sizes[0]==sizes[1])