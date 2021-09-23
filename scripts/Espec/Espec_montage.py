#Espec_montage.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

# start with an example shot - same workflow as used in LivePlotting


def create_montage(image, E_roi, y_roi, downsample_E=1, downsample_y=1):
    count, m, n = image.shape
    #mm = int(ceil(sqrt(count)))
    #nn = mm
    
    m = y_roi[1] - y_roi[0]
    n = E_roi[1] - E_roi[0]
    
    # m is energy axis, n is y axis
    m = int(m /  downsample_y)
    n = int(n / downsample_E)
    
    
    mm=count
    nn=1
    M = np.zeros((nn * n, mm * m))
    x_ax=np.linspace(0, m*(mm-1), count)+m/2.0

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = image[image_id, y_roi[0]:y_roi[1]:downsample_y, E_roi[0]:E_roi[1]:downsample_E].T
            image_id += 1
    return M, x_ax

#%%

diag = 'espec1'
charge=-1.0
#date='20210618'
#run= 'run18'

date='20210621'
run= 'run20'

#### replace DATA_FOLDER ####
run_name = date+'/run99'
spec = Espec(run_name,shot_num=1,img_bkg=None,diag=diag)




#%%
# data pipeline : code for whole run analysis
eSpec_spec_pipe = DataPipeline(diag, spec.eSpec_proc.espec_data2screen, single_shot_mode=True)

#%%
# Analysis

shot_num, data = eSpec_spec_pipe.run('%s/%s'%(date, run))


#%%
#spec_counts_per_mrad_whole_run=data*eSpec_proc.eAxis_MeV*distance2screen_dict[diag]*1e-3
# y_mrad = (eSpec_proc.screen_y_mm-38)/distance2screen_dict[diag]# -38 to centre mrad scale on the screen
# create and plot montage of all shots in a run

# add in downsamping otherwise image too big for my CPU too handle!

downsample_E = 5
downsample_y = 5

E_roi = (0, data.shape[2])
y_roi = (0, data.shape[1])

#E_roi = (1200, 4000) # in pixels
#y_roi = (360, 560) # in pixels

montage, x_ax=create_montage(data, E_roi, y_roi, downsample_E, downsample_y)

#%%

from matplotlib import cm

brightness_scale = np.percentile(montage, 99.99)

plt.figure(figsize=(10,6))
ax = plt.gca()
#ax.xaxis.set_label_position('top')
#ax.xaxis.tick_top()
#ax.set_xticks(x_ax[::3])
#ax.set_xticklabels(shot_num[::3])
#ax.set_xlabel('shot number')
ax.set_xticks([])

xaxis = spec.eSpec_proc.eAxis_MeV
xaxis = xaxis[::-1]
xaxis = xaxis[E_roi[0]:E_roi[1]:downsample_E]


im=ax.pcolormesh(np.arange(montage.shape[1]), xaxis, montage, vmin=0.0, vmax=brightness_scale)
ax.set_ylabel(r'$E$ [MeV]')
ax.set_title("%s %s"%(date, run), y=-0.1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cb=plt.colorbar(im, cax=cax)
cb.set_label(r'Ed$^2$counts/d$\theta$d$E$ [counts mrad$^{-1}$]')
plt.tight_layout()

N = data.shape[0]
x_bounds = x_ax[:-1] + np.diff(x_ax)/2.0
#[ax.axvline(x=x0, color='w') for x0 in x_bounds]

plt.show()