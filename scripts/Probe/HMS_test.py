#HMS_test.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

#date = '20210621'
#run = 'run04'

date = '20210602'
run = 'run21'

shot = 71 #shot='Shot002'
file_ext = '.TIFF'

# get the analysis object
diag = 'HMSS'
run_name = date+'/' + 'run11'
HMS = Interferometry(run_name,shot_num=1,diag=diag,cal_data_path=None)

#%%
# grab a shot 
l = [date, run, shot]
filepath = HMS.get_filepath(l)
#filepath = ROOT_DATA_FOLDER + '/' + diag + '/' +  date + '/' + run +'/' + shot + file_ext
im = HMS.get_raw_image(filepath)



bkg_l = [date, run, 1]
bkg_filepath = HMS.get_filepath(bkg_l)
bkg_im = HMS.get_raw_image(bkg_filepath)

im_r = im/bkg_im 
im_r -= np.mean(np.isfinite(im_r))
im_r[bkg_im==0.0] = 0.0

from scipy.ndimage import median_filter, gaussian_filter
o = 3
im_r =  median_filter(im_r, o)

im_r = gaussian_filter(im_r, o)

vmin = np.percentile(im_r, 0.5)
vmax = np.percentile(im_r, 99.5)

plt.figure()
plt.imshow(im_r, vmin=vmin, vmax=vmax, origin='lower')

#plt.figure()
nrows, ncols = im_r.shape
x,y = np.arange(ncols), np.arange(nrows)

peaks = []
troughs = []
for i in range(20, nrows, 50):
    sub_im = im_r[i-20:i+20, :]
    lineout = np.mean(sub_im, axis=0)
    lineout -= np.mean(lineout)
    lineout *= 50
    yi =  i
    plt.plot(x, lineout+yi)
    plt.axhline(y=yi, color='k', alpha=0.25)
    peaks.append(x[np.argmax(lineout)])
    troughs.append(x[np.argmin(lineout)])

py = [i for i in range(20, nrows, 50)]
plt.plot(peaks, py, color='w', marker='.')
plt.plot(troughs, py, color='w', marker='.')

#%%
# grab many shots

# not sure about 108
gd_shots = [59, 58] + list(range(31,43+1)) + list(range(20,29+1)) + list(range(102,114+1)) +  list(range(79,98+1)) + list(range(66,74+1)) + list(range(146,165+1)) + list(range(136,144+1)) + [134] 
gd_shots.sort()
gd_shots.remove(163)
gd_shots.remove(160)
gd_shots.remove(138)
gd_shots.remove(93)
gd_shots.remove(67)
gd_shots.remove(108)

var = []
for shot in range(1,171+1):
    if shot in gd_shots:
        l = [date, run, shot]
        filepath = HMS.get_filepath(l)
        #filepath = ROOT_DATA_FOLDER + '/' + diag + '/' +  date + '/' + run +'/' + shot + file_ext
        im = HMS.get_raw_image(filepath)    
        plt.figure()
        vmin = np.percentile(im, 0.5)
        vmax = np.percentile(im, 99.5)
        
        plt.imshow(im, vmin=vmin, vmax=vmax)
        plt.title(shot)