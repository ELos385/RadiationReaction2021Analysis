#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clicker approach instead

Created on Sat Oct  2 13:32:25 2021 by Cary Colgan. 
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

from lib.figure_clicker import list_clicks_on_ax
from modules.Espec.espec_wire_tracking import get_shadows_from_clicks



# start with an example shot - same workflow as used in LivePlotting

#date = '20210618'
date = '20210618'
run = 'run02'

espec_diag_names = ['espec1', 'espec2']
file_ext = '.tif'


# make objects for holding espec analysis data
specs = []
for n, espec_diag_name in enumerate(espec_diag_names):
    run_name = date+'/run99'

    spec = Espec(run_name,shot_num=1,img_bkg=None,diag=espec_diag_name)
    specs.append(spec)
    
#%%

shot='Shot049'

n = 13
wire_width=6.0

# grab a shot's images - like LivePlotting would do 
fig, axes = plt.subplots(2,1)
fig.suptitle('%s %s %s' % (date, run, shot))
#fig2, axes2 = plt.subplots(2,1, sharex=True)


screen_distances = [1760.0, 2300.0] # according to Matt's tForm files

data = []
for idx, spec in enumerate(specs): 
    filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot + file_ext
    #filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot
    im = spec.get_image(filepath).T
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
    
#%%
dicto = data[0]
x_mm, y, im, x_MeV = np.copy(dicto['x_mm']), np.copy(dicto['y']), np.copy(dicto['im']), np.copy(dicto['x_MeV'])
    

y = np.nanmean(im, axis=0)

x = np.copy(x_mm)
y = np.copy(y)


# do for espec1
fig, ax = plt.subplots()
ax.plot(x, y)
click, espec1_list = list_clicks_on_ax(ax)

# these peaks will then in coordinates of mm



#%%
list_of_rough_peaks1 = [
 88.0,
 100.87880463059312,
 112.88948686264307,
 119.58831945889696,
 125.98265966441207,
 137.8578629032258,
 150.34205092351715,
 162.2172541623309,
 169.2205791493236,
 175.00593457336106,
 188.09910737513005,
 200.58329539542143,
 213.6764681971904,
 219.7663160119667,
 226.16065621748174]


#list_of_rough_peaks1 = [i[0] for i in espec1_list]

list_of_rough_peaks1 = np.array(list_of_rough_peaks1)
list_of_rough_peaks1 = list_of_rough_peaks1[::-1][range(n)]
list_of_rough_peaks1 = list_of_rough_peaks1[::-1]
shadows1 = get_shadows_from_clicks(x, y, list_of_rough_peaks1, 
                            wire_width=2.8, smoothing_factor=1e-5, 
                            plotting=True)


#%%
dicto = data[1]
x_mm, y, im, x_MeV = np.copy(dicto['x_mm']), np.copy(dicto['y']), np.copy(dicto['im']), np.copy(dicto['x_MeV'])
    
y = np.nanmean(im, axis=0)

x = np.copy(x_mm)
y = np.copy(y)


#repeat for espec2
fig, ax = plt.subplots()

ax.plot(x, y)
click, espec2_list = list_clicks_on_ax(ax)


# these peaks will then in coordinates of mm

#%%
list_of_rough_peaks2 = [
 66.0,
 86.12903225806451,
 104.17859001040581,
 113.62187825182102,
 123.49440686784598,
 142.38098335067636,
 161.26755983350674,
 179.72489594172734,
 190.02666493236208,
 198.61147242455772,
 217.92728928199787,
 237.67234651404783,
 256.5589229968782,
 266.860691987513,
 275.87473985431836]

#list_of_rough_peaks2 = [i[0] for i in espec2_list]
list_of_rough_peaks2 = np.array(list_of_rough_peaks2)
list_of_rough_peaks2 = list_of_rough_peaks2[::-1][range(n)]
list_of_rough_peaks2 = list_of_rough_peaks2[::-1]

shadows2 = get_shadows_from_clicks(x, y, list_of_rough_peaks2, 
                            wire_width=wire_width, smoothing_factor=0.5e-5, 
                            plotting=True)

#%%

shadow_sets = [shadows1, shadows2]

if shadows1.shape[0] != shadows2.shape[0]:
    # not found same on each screen
    pass

else:

    save_q = input('Save (y or n): ')
     
    
    for idx, s in enumerate(shadow_sets):
        shadows = shadow_sets[idx]
        diag = 'espec' + str(idx+1)
        
        ss = shot.split('.')[0]
        
        dicto_to_save = {'date': date,
                         'run': run,
                         'ss': shot,
                         'diag': diag,
                         'date_analysed': '20211002',
                         'analysis_kwargs': 'clicker',
                         'shadows': shadows,
                         'script': '20211001_Espec_wire_tracking_clicker_test.py'
                         }
    
        fp = HOME + '/results/' + 'Espec/' + 'wire_shadows_20211001/' + diag +'/'
        fn = diag + '_' + date + '_' + run + '_' + ss + '_wire_shadows.pkl'
        
        if save_q=='y':
            save_object(dicto_to_save, fp + fn)
            plt.close('all')