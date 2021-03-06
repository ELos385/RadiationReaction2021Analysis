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

plt.style.use('/Users/ee.los/.matplotlib/mpl_configdir/thesis.mplstyle')

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

diag = 'espec1'
from espec_ref_files import espec1_ref_files, espec2_ref_files
espec_ref_files = espec1_ref_files
ref_files = []

date = '20210427'


for run in espec_ref_files[date]:
    # run is then a dict
    run_names = list(run.keys())
    for rn in run_names:
        if rn=='filepath':
            # points to actual calib files
            shot_list = run[rn]
            shot_list = [ROOT_DATA_FOLDER + '/../calibrations/' + diag + '/' + shot for shot in shot_list]
            ref_files += shot_list
        else:
            # points to shots from Mirage data folder
            shot_list = run[rn]
            shot_list = [ROOT_DATA_FOLDER + '/' + diag + '/' + date + '/' + rn + '/' + shot for shot in shot_list]
            ref_files += shot_list


ref_files = np.array([imread(ref_file) for ref_file in ref_files])
img = np.mean(ref_files, axis=0)

imgP_pix, imgP_real = [[]], [[]]

# mark on points - comment out if unsure
# first is (x,y) in pixels, and second is (x, y) physcially on ruler in mm

imgP_pix_real = np.array([[1243, 104], [10, 0],
                     [1177, 1019], [150, 0],
                     [1111, 2000], [290, 0],
                     [1588, 1840], [270, 76.5],
                     [1635, 970], [150, 76.5],
                     [1693, 10], [10, 76.5]
                     ])

imgP_pix, imgP_real =  imgP_pix_real[::2], imgP_pix_real[1::2]

plt.figure()
im1=plt.imshow(img)#, vmax=10000)
plt.title('%s' % (date))
plt.plot(imgP_pix[:,0],imgP_pix[:,1],'r+')
plt.xlabel('Pixels in x')
plt.ylabel('Pixels in y')
cb=plt.colorbar(im1)
cb.set_label('Counts on CCD', rotation=270, labelpad=20)
plt.tight_layout()
plt.show()

#%%

# size of physical range covered in picture
# nominal values to cover the lanex only would be to
xRange, yRange = 330.0, 200.0#300, 76.5

#xRange =400
#yRange = 106

# size of new img - can upsample range to beyond img pixels if you wanted
Nx_new = 4000
Ny_new = 960

# spacing between edge of lanex screen and edge of physical range requested
# if you have just requested the lanex screen, then nominal values would be
x0, y0 = -10, -50#0, 0

#x0 = -50
#y0 = -15

# physical resolution of output image
dx = xRange/Nx_new
dy = yRange/Ny_new
x_mm = x0 + np.linspace(0,Nx_new,num=Nx_new)*dx
y_mm = y0 + np.linspace(0,Ny_new,num=Ny_new)*dy

# Pixel values in new image (given any upsampling or offsets)
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

# plt.figure()
# #im_out2=espec_warp(img, tForm)
# im=plt.imshow(im_out)#,extent= (np.min(x_mm), np.max(x_mm), np.max(y_mm), np.min(y_mm)))
# #plt.plot(imgP_real[:,0],imgP_real[:,1],'r+')
# #plt.title('%s- Transformed' % (date))
# plt.colorbar(im)
# plt.show()

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
im=plt.imshow(im_out2,extent= (np.min(x_mm), np.max(x_mm), np.max(y_mm), np.min(y_mm)))
plt.plot(imgP_real[:,0],imgP_real[:,1],'r+')
plt.title('%s- Transformed' % (date))
cb=plt.colorbar(im)
cb.set_label('Counts on CCD', rotation=270, labelpad=10)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.tight_layout()

#%%
# save transform file if happy

# desired tform details
# fname_diag = diag #+ '_big'
# date = date
# run = 'run08'
# shot = 'shot004'
#
# tForm_filedir = HOME + '/calib/' + diag + '/'
# tForm_filename = '%s_transform_%s_%s_%s.pkl' % (fname_diag, date, run, shot)
# tForm_filepath = tForm_filedir + tForm_filename
#
# check = input("Save tform file as %s \nin %s \ny or n? " % (tForm_filename, tForm_filedir))
#
# if check=='y':
#     save_object(tForm, tForm_filepath)
#     print('saved: ', tForm_filepath)
# else:
#     print('not saved.')


#%%
plt.show()
