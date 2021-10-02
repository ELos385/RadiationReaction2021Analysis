#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script iterates the find_wire_shadows algorithm over a run of shots.
This is done for a fixed set of analysis parameters in the hope that the 
over the vast number of shots it is bound to work for some.

Function plots the results as a suspected 'good fitting' which if demmed true,
the shadow data can be saved by the final part of the script.

A good fitting is one where the wires are appropriately spaced to know none have
been missed, and where there are the same number of wires identified on both 
espec1 and espec2

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
from os import listdir    

all_files = listdir(ROOT_DATA_FOLDER + '/' + 'espec1' + '/' +  date + '/' + run +'/')
all_files = [i for i in all_files if i[-4:]=='.tif']

start_shot='Shot004'

s_idx = all_files.index(start_shot + '.tif')

if s_idx==0: 
    s_idx=0
else:
    s_idx = s_idx + 1


for shot in all_files[s_idx:]:
    
    try:
    
        # loop analysis over 
        
        # grab a shot's images - like LivePlotting would do 
        fig, axes = plt.subplots(2,1)
        #fig2, axes2 = plt.subplots(2,1, sharex=True)
        
        screen_distances = [1740.0, 2300.0]
        
        data = []
        for idx, spec in enumerate(specs): 
            #filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot + file_ext
            filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot
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
        
        shadow_sets = []
        for idx, dicto in enumerate(data):
            #x,y,im = dicto['x_MeV'], dicto['y'], dicto['im']
            x_mm, y, im, x_MeV = np.copy(dicto['x_mm']), np.copy(dicto['y']), np.copy(dicto['im']), np.copy(dicto['x_MeV'])
            
            shadows = find_wire_shadows(x_mm, im, plotting=True, **analysis_kwargs)
            
                
            E_vals = np.interp(shadows[:,0], x_mm, x_MeV)
            
            ax.errorbar(x=shadows[:,0], y=E_vals, xerr=shadows[:,2], linestyle='', marker='x', capsize=2, label='espec%g' % (idx+1))
            sizes.append(shadows[:,0].size)
            
            shadow_sets.append(shadows)
        
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('T($\\theta=0$) [MeV]')
        ax.grid()
        ax.legend()
        
        #print('Same number on both screens?: ', sizes[0]==sizes[1])
        
        flag = True
        
        if sizes[0]!=sizes[1]:
            # inconsistent between screens - ignore
            flag = False
        
        # notice a pattern
        
        firsts = np.diff(np.sort(shadow_sets[0][:,0]))
        seconds = np.diff(np.sort(shadow_sets[1][:,0]))
        lb1, ub1, lb2, ub2 = 5, 7, 11, 13
        ids  = ((firsts >= lb1) & (firsts <= ub1)) | ((firsts >= lb2) & (firsts <= ub2))
        test1 = firsts[ids]
        test1 = len(test1)==len(firsts)
        
        lb1, ub1, lb2, ub2 = 8, 10, 17, 19
        ids  = ((seconds >= lb1) & (seconds <= ub1)) | ((seconds >= lb2) & (seconds <= ub2))
        test2 = seconds[ids]
        test2 = len(test2)==len(seconds)
        
        final_flag = flag & test1 & test2
        
        if final_flag==True:        
            print('found one at %s' % shot)
            break
        else:
            #print('bad recognitiion')
            plt.close('all')
        
    
    except(FileNotFoundError, IndexError):
        plt.close('all')
        pass

if shot==all_files[-1]:
    print('finished searching %s %s' % (date, run))

#%%
for idx, s in enumerate(shadow_sets):
    shadows = shadow_sets[idx]
    diag = 'espec' + str(idx+1)
    
    ss = shot.split('.')[0]
    
    dicto_to_save = {'date': date,
                     'run': run,
                     'ss': shot,
                     'diag': diag,
                     'date_analysed': '20210916',
                     'analysis_kwargs': analysis_kwargs,
                     'shadows': shadows,
                     'script': '20210916_Espec_wire_tracking_test.py'
                     }

    fp = HOME + '/results/' + 'Espec/' + 'wire_shadows/' + diag +'/'
    fn = diag + '_' + date + '_' + run + '_' + ss + '_wire_shadows.pkl'
    # save_object(dicto_to_save, fp + fn)