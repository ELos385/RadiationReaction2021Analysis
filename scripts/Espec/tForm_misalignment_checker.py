#tForm_misalignment_checker.py
"""
idea is to make a script that can check if tForm wrong (likely because espec
camera was knocked)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cv2
from scipy.sparse import diags

import sys
if '../../' not in sys.path: sys.path.append('../../')

from lib.general_tools import *
from setup import *
from modules.Espec.Espec import *

def espec_warp(img, tform):
    """written so can apply any tForm 
    """
    imgArea0=tform['imgArea0']
    H=tform['H']
    newImgsize=tform['newImgSize']
    imgArea1=tform['imgArea1']
    img=img-np.median(img)
    img=sig.medfilt2d(img, kernel_size=(3, 3))
    imgCountsPerArea=img/imgArea0
    imgCountsPerArea[imgArea0==0]=0
    imgCountsPerArea[np.isinf(imgCountsPerArea)]=0
    imgCountsPerArea[np.isnan(imgCountsPerArea)]=0
    img_out=cv2.warpPerspective(imgCountsPerArea, H, newImgsize)*imgArea1
    return img_out




#%%
# inspect a shot

diag = 'espec2'
date = '20210614'
run = 'run12'
shot = 'Shot006'
file_ext = '.tif'

# make spec obj - this will grab what should be the correct tForm file
spec = Espec(date+'/'+run,shot_num=1,img_bkg=None,diag=diag)

#tForm = load_object(spec.tForm_filepath)

filepath = ROOT_DATA_FOLDER + '/%s/%s/%s/%s%s' % (diag, date, run, shot, file_ext)
img = imread(filepath)

tForm_filepath = choose_cal_file(date+'/'+run, shot, diag, diag+'_transform', cal_data_path=HOME + '/calib/')

# check transform is correct by seeing whole image
tForm = load_object(tForm_filepath)


# plot whole image
plt.figure()

im_out2 = espec_warp(img, tForm)

x_mm = tForm['x_mm']
y_mm = tForm['y_mm']
plt.imshow(im_out2,extent=(x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()),
           vmax=10)
x_low, x_high = 0.0, 250.0
y_low, y_high = 0.0, 59.0
plt.plot([x_low, x_low, x_high, x_high], [y_low, y_high, y_high, y_low], 'r+')
plt.title('Check:  %s %s %s' % (date, run, shot))

#%%
# inspect a whole load of shots to check if its gone wrong
from glob import glob

head = ROOT_DATA_FOLDER + '/' + diag +'/'
all_dates = glob(head+'*/')
all_runs = [glob(i+'*/') for i in all_dates]

all_dates_clipped = [i[len(head):-1] for i in all_dates]
all_runs_clipped = [[r[len(head)+len(d)+1:-1] for r in all_runs[idx]] for idx, d in enumerate(all_dates_clipped)]
file_ext = '.tif'

all_dates = all_dates_clipped
all_runs = all_runs_clipped

date_to_choose = '20210608'

vmax= 10
# iterate over all shots that day - plot for a random shot
d_idx = all_dates.index(date_to_choose)
for r in all_runs[d_idx]:
    shots = glob('%s%s/%s/*%s' % (head, date_to_choose, r, file_ext))
    
    if len(shots)==0:
        pass
    else:
        filepath = np.random.choice(shots)
        shot = filepath[len(head)+len(date_to_choose)+len(r)+2:-len(file_ext)]
        
        plt.figure()
        img = imread(filepath)
        im_out2 = espec_warp(img, tForm)
        plt.imshow(im_out2,extent=(x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()),
               vmax=vmax)
        plt.plot([x_low, x_low, x_high, x_high], [y_low, y_high, y_high, y_low], 'r+')
        plt.title('Check:  %s %s %s' % (date_to_choose, r, shot))

plt.show()