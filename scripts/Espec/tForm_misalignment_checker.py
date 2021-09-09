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
#from modules.Espec.Espec import *

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
date = '20210513'
run = 'run09'
shot = 'Shot006'
file_ext = '.tif'

# make spec obj - this will grab what should be the correct tForm file
spec = Espec(date+'/'+run,shot_num=1,img_bkg=None,diag=diag)

tForm = load_object(spec.tForm_filepath)

filepath = ROOT_DATA_FOLDER + '/%s/%s/%s/%s%s' % (diag, date, run, shot, file_ext)
img = imread(filepath)


# check transform is correct by seeing whole image

tForm = load_object(spec.tForm_filepath)

x,y = tForm['x_mm'], tForm['y_mm']

dx, dy = np.gradient(x)[0], np.gradient(y)[0]
x_new = np.array(list(x-x.max()) + list(x))
y_new = np.array(list(y-y.max()) + list(y))

tForm['newImgSize'] = tForm['newImgSize']*2
tForm['x_mm'] = x_new
tForm['y_mm'] = y_new

plt.figure()

im_out2 = espec_warp(img, tForm)

x_mm = tForm['x_mm']
y_mm = tForm['y_mm']
plt.imshow(im_out2,extent= (np.min(x_mm), np.max(x_mm), np.max(y_mm), np.min(y_mm)), 
           vmax=100)
