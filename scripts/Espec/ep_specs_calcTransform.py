#ep_specs_calcTransform.py

import sys, os, pickle
if '../../' not in sys.path: sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cv2
from scipy.sparse import diags

from lib.general_tools import *
from setup import *

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
# CC (18-05-21) edit on Jupyter Notebook

# get a reference shot to see where physical values (ruler markings) are 
# relative to pixels in captured image
    
# mark points for calculating transform
    
diag = 'espec2'

date = '20210513'
run = 'ref02'
shot = 'Shot001'
ref_file = ROOT_DATA_FOLDER + '/' + diag + '/' + date + '/' + run + '/' + shot + '.tif'
imgP_pix, imgP_real = [[]], [[]]


# mark on points - comment out if unsure
imgP_pix = np.array([[203,853], [1367, 878], [2545, 903], [2465, 1327], [1528, 1306], [235, 1273]]) # pixel values
imgP_real =np.array([[1, 59], [140, 59], [280, 59], 
                     [275, 0], [160, 0], [1, 0]]
                     ) # physical values (i.e. in cm)

plt.figure()
img = imread(ref_file)
plt.imshow(img)
plt.title('%s %s %s' % (date, run, shot))
plt.plot(imgP_pix[:,0],imgP_pix[:,1],'r+')

#%%


xRange =400
yRange = 106


Nx_new = 4000
Ny_new = 960

# spacing between edge of lanex screen and image edge
x0 = -50
y0 = -15

# physical resolution of output image
dx = xRange/Nx_new
dy = yRange/Ny_new
x_mm = x0 + np.linspace(0,Nx_new,num=Nx_new)*dx
y_mm = y0 + np.linspace(0,Ny_new,num=Ny_new)*dy

# Pixel values in new image
imgP_trans = (imgP_real-[x0,y0]) / [dx,dy]

H, status = cv2.findHomography(imgP_pix,imgP_trans)

#%%
# make transform file

(Ny,Nx) = np.shape(img)

# calculate pixel areas in original image
retval,H_inv = cv2.invert(H)
(X,Y) = np.meshgrid(x_mm,y_mm)
X_raw = cv2.warpPerspective(X, H_inv, (Nx,Ny))
Y_raw = cv2.warpPerspective(Y, H_inv, (Nx,Ny))
imgArea0 = np.abs(np.gradient(X_raw,axis=1)*np.gradient(Y_raw,axis=0))

# calc transformed image
imgCountsPerArea = img/imgArea0
# imgCountsPerArea[imgArea0==0] =0
# imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
# imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

im_out = cv2.warpPerspective(imgCountsPerArea, H, (Nx_new,Ny_new))*dx*dy

tForm = {  
    'description': ' image transform',
    'H': H,
    'newImgSize':(Nx_new,Ny_new),
    'x_mm': x_mm,
    'y_mm': y_mm,
    'imgArea0': np.median(imgArea0[np.abs(X_raw**2+Y_raw**2)>0]),
    'imgArea1': dx*dy,
}

#%%
# check transform is correct

plt.figure()
im_out2=espec_warp(img, tForm)
plt.imshow(im_out2,extent= (np.min(x_mm), np.max(x_mm), np.max(y_mm), np.min(y_mm)))
plt.plot(imgP_real[:,0],imgP_real[:,1],'r+')


#%%
# save transform file if happy

# desired tform details
date = date
run = 'run01'
shot = 'shot001'

tForm_filedir = HOME + '/calib/' + diag + '/'
tForm_filename = '%s_transform_%s_%s_%s.pkl' % (diag, date, run, shot)
tForm_filepath = tForm_filedir + tForm_filename

check = input("Save tform file as %s \nin %s \ny or n? " % (tForm_filename, tForm_filedir))

if check=='y':
    save_object(tForm, tForm_filepath)
    print('saved: ', tForm_filepath)
else:
    print('not saved.')


#%%
plt.show()