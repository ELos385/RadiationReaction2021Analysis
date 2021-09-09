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
fig, axes = plt.subplots(2,1)

fig2, axes2 = plt.subplots(2,1, sharex=True)

for idx, spec in enumerate(specs): 
    filepath = ROOT_DATA_FOLDER + '/' + spec.diag + '/' +  date + '/' + run +'/' + shot + file_ext
    im = spec.get_image(filepath)
    x, y = spec.x_mm, spec.y_mm
    
    left,right,bottom,top = x.min(), x.max(), y.min(), y.max()
    extent = left,right,bottom,top
    axes[idx].imshow(im)#, extent=extent)
    if hasattr(spec, 'p_labels'):
        for p in np.array(spec.xaxis)[:,0]:
            #win.docks[spec].widgets[0].view.addLine(x=p) 
            axes[idx].axvline(x=p, color='w', alpha=0.5)
    

    # try plotting w.r.t energy axis instead
    x, y = spec.x_MeV, spec.y_mm
    cols = ~np.isnan(x)
    axes2[idx].pcolormesh(x[cols], y, im[:, cols])

axes[1].set_xlabel('x [pixels]')

axes2[0].set_xlim((1500, 350))
print('finished')


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