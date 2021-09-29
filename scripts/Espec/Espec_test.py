#Espec_test.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *

# start with an example shot - same workflow as used in LivePlotting

date = '20210618'
run = 'run18'
shot='Shot020'

espec_diag_names = ['espec1', 'espec2']
file_ext = '.tif'


# make objects for holding espec analysis data
specs = []
for n, espec_diag_name in enumerate(espec_diag_names):
    run_name = date+'/run99'

    spec = Espec(run_name,shot_num=1,img_bkg=None,diag=espec_diag_name)
    specs.append(spec)

#%%

# grab a shot's images - like LivePlotting would do 
fig, pixel_axes = plt.subplots(2,1)

fig2, MeV_axes = plt.subplots(3,1, sharex=True)

screen_distances = [1740.0, 2300.0]

for idx, spec in enumerate(specs): 
    filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot + file_ext
    im = spec.get_image(filepath).T
    x, y = spec.x_mm, spec.y_mm
    
    left,right,bottom,top = x.min(), x.max(), y.min(), y.max()
    extent = left,right,bottom,top
    pixel_axes[idx].imshow(im)#, extent=extent)
    if hasattr(spec, 'p_labels'):
        for p_idx in range(len(spec.xaxis)):
            p, p_E = spec.xaxis[p_idx]
            
            pixel_axes[idx].text(p, 0, '%.1f' % (p_E*1e-3),
                verticalalignment='top', horizontalalignment='center', color='w')
            #win.docks[spec].widgets[0].view.addLine(x=p) 
            pixel_axes[idx].axvline(x=p, color='w', alpha=0.5)
    

    # try plotting w.r.t energy axis instead
    x, y = spec.x_MeV, spec.y_mm
    y = y/screen_distances[idx]
    
    y -= np.nanmean(y)
    
    cols = ~np.isnan(x)
    MeV_axes[idx].pcolormesh(x[cols], y, im[:, cols])
    
    # 1d lineouts like LivePlotting
    espec_raw_img = spec.get_raw_image(filepath)
    
    # d() from genreal tools, avg difference across axis - dy is constant after homogrpahic tForm anwyay
    espec_spectrum = np.sum(spec.eSpec_proc.energy_spectrum(espec_raw_img),axis=0)*d(spec.eSpec_proc.screen_y_mm)
    x = spec.eSpec_proc.eAxis_MeV
    espec_spectrum /= np.nanmax(espec_spectrum), 
    MeV_axes[-1].plot(x, espec_spectrum)

pixel_axes[-1].set_xlabel('x [tFormed pixels]')
MeV_axes[-1].set_xlabel('T [MeV]')

#axes2[0].set_xlim((1500, 350))
print('finished')

#%%
# check 1D lineouts funcs from LivePlotting

"""
def return_electron_spectra(espec_file_list):
    espec1_filepath = espec_file_list[0]
    espec2_filepath = espec_file_list[1]

    espec1_img = spec_obj[0].get_raw_image(espec1_filepath)
    espec1_spectrum = np.sum(spec_obj[0].eSpec_proc.energy_spectrum(espec1_img),axis=0)*d(spec_obj[0].eSpec_proc.screen_y_mm)
    
    espec2_img = spec_obj[1].get_raw_image(espec2_filepath)

    espec2_spectrum = np.sum(spec_obj[1].eSpec_proc.energy_spectrum(espec2_img),axis=0)*d(spec_obj[1].eSpec_proc.screen_y_mm)
    return ((spec_obj[0].eSpec_proc.eAxis_MeV, espec1_spectrum),(spec_obj[1].eSpec_proc.eAxis_MeV, espec2_spectrum))
espec_colors = [(255,100,100),(100,100,255)]
win.add_multiline_plot('electron_spectra', ['espec1','espec2'], return_electron_spectra,line_colors=espec_colors)
"""



#%%
# test out a function on a whole run
charge = []
for spec in specs:
    espec_pipeline = DataPipeline(spec.diag, spec.get_total_charge_from_im, single_shot_mode=True)

    shot_num, charge_data = espec_pipeline.run('%s/%s'%(date, run))
    charge_data=np.array(charge_data)
    charge.append(charge_data)
    
charge = np.array(charge).T

#%%
# plot charge for the run
plt.figure()
plt.plot(shot_num, charge, '.-')
plt.title('%s/%s'%(date, run)), plt.xlabel('Shot #'), plt.ylabel('Total Charge [pC]')
plt.grid()

#%%
