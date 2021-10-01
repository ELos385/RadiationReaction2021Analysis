#ep_specs_calcTransform.py
"""
Check details about espec2 screen from calib photos
"""
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
fp = ROOT_DATA_FOLDER + '/../calibrations/' + diag + '/' + '20210622_20_second_exposure.tiff'

img = imread(fp)
imgP_pix, imgP_real = [[]], [[]]

# mark on points - comment out if unsure
# first is (x,y) in pixels, and second is (x, y) physcially on ruler in mm

imgP_pix_real = np.array([[220, 1281], [10, 0],
                     [1361, 1306], [150, 0],
                     [2507, 1328], [280, 0],
                     [2461, 902], [280, 59],
                     [1365, 880], [150, 59],
                     [271, 859], [20, 59]
                     ]
                    )

imgP_pix, imgP_real =  imgP_pix_real[::2], imgP_pix_real[1::2]

plt.figure()
plt.imshow(img, vmax=1000)
plt.title('%s' % (date))
plt.plot(imgP_pix[:,0],imgP_pix[:,1],'r+')


#%%
fp = ROOT_DATA_FOLDER + '/../photos/' + 'RR2021 Cary Phone Photos/' + 'IMG_20210428_091203.jpg'

img = imread(fp)
img = rgb2gray(img)
imgP_pix, imgP_real = [[]], [[]]

# mark on points - comment out if unsure
# first is (x,y) in pixels, and second is (x, y) physcially on ruler in mm

imgP_pix_real = np.array([[1234, 976], [280, 0],
                     [1259, 2387], [150, 0],
                     [1280, 3847], [10, 0],
                     [1914, 3787], [10, 59],
                     [1892, 2064], [180, 59],
                     [1875, 791], [300, 59]
                     ]
                    )

imgP_pix, imgP_real =  imgP_pix_real[::2], imgP_pix_real[1::2]

plt.figure()
plt.imshow(img, vmax=1000)
plt.title('%s' % (date))
plt.plot(imgP_pix[:,0],imgP_pix[:,1],'r+')


#%%

#img = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
# size of physical range covered in picture
# nominal values to cover the lanex only would be to
xRange, yRange = 300, 76.5 

xRange =500
yRange = 76 + 100

# size of new img - can upsample range to beyond img pixels if you wanted
Nx_new = 4000
Ny_new = 960

# spacing between edge of lanex screen and edge of physical range requested
# if you have just requested the lanex screen, then nominal values would be 
x0, y0 = 0, 0

x0 = -100
y0 = -50

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
plt.imshow(im_out2,extent= (np.min(x_mm), np.max(x_mm), np.max(y_mm), np.min(y_mm)), vmax=5e16)
plt.plot(imgP_real[:,0],imgP_real[:,1],'r+')
plt.title('%s- Transformed' % (date))


#%%
plt.show()

#%%
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


def ruler_marks_func(x,A,c,x0, sigma_mm):
    """ideal ruler markings profile, but unresolved by sigma_mm factor
    """
    # upsample x
    x_new = np.linspace(x.min(), x.max(), 10*x.size)
    
    dx = 0.1 # markings are 100 um wide
    ideal = np.zeros_like(x_new)
    points = np.arange(int(x_new.min()), int(x_new.max())+1) + x0
    lb_set = points - dx/2.0
    ub_set = points + dx/2.0
    for lb,ub in zip(lb_set,ub_set):
        ids = (x_new>=lb) & (x_new<=ub)
        ideal[ids] = 1.0
    
    dx_dp = np.mean(np.gradient(x_new))
    sigma_pix = sigma_mm / dx_dp
    y_prof = gaussian_filter(ideal, sigma_pix)
    output = c + A*y_prof
    
    output_reg_sample = np.interp(x, x_new, output)
    return output_reg_sample


def find_spatial_resolution_from_lineout(x, y, dx_dp, smooth_troughs=0.001,
                                         smooth_peaks=0.001, 
                                         plotter=True):    
    plt.figure()
    plt.title('Trough fitting')
    plt.plot(x, y)
    troughs, _ = find_peaks(-y, distance=0.75/dx_dp)
    plt.plot(x[troughs], y[troughs], "x")
    
    spl = UnivariateSpline(x[troughs], y[troughs])
    spl.set_smoothing_factor(smooth_troughs)
    plt.plot(x, spl(x), 'g')
    
    y = y-spl(x)
    
    plt.figure()
    plt.title('Peak fitting')
    plt.plot(x,y)
    peaks, _ = find_peaks(y, distance=0.75/dx_dp)
    plt.plot(x[peaks], y[peaks], "x")
    spl = UnivariateSpline(x[peaks], y[peaks])
    spl.set_smoothing_factor(smooth_peaks)
    plt.plot(x, spl(x), 'g')
    
    y /= spl(x)
    
    
    plt.figure()
    plt.plot(x,y)
    plt.title('Markings properly normalised')
     

    plt.figure()
    plt.plot(x, y)
    plt.title('Fitted markings')
    
        
    p0 = [3.0, 0.00, 0.0, 0.115]
    bounds = [[0.4, 10.0], [-0.001, +0.001], [-0.5, +0.5], [0.05, 0.450]]
    bounds = list(zip(*bounds))
    
    popt,pcov = curve_fit(ruler_marks_func, x, y, p0, bounds=bounds)
    plt.plot(x, ruler_marks_func(x, *popt), 'r--')
    
    plt.plot(x, ruler_marks_func(x, 1.0, popt[1], popt[2], 0.0), 'k')
    
    perr = np.diagonal(pcov)**(0.5)
    
    print('Resolution: ', 1e3*popt[-1], 'um ± ', (1e2*perr/popt)[-1], '%')

    return popt, perr


# check resolution method across lower ruler markings
ymin, ymax = 60.5, 63.5
xmin, xmax = 1.5, 300.0

y_ids = (y_mm >= ymin) & (y_mm <= ymax)
x_ids = (x_mm >= xmin) & (x_mm <= xmax)

sub_img = im_out2[:, x_ids]
sub_img = sub_img[y_ids, :]

x = x_mm[x_ids]
ruler_marks = np.nanmean(sub_img, axis=0)

plt.figure()
plt.plot(x, ruler_marks)
N = 100
run_mean = np.convolve(ruler_marks, np.ones(N)/N, mode='same')
plt.plot(x, run_mean, 'r--')
#plt.figure()
#plt.plot(x, ruler_marks - run_mean)

# take a smaller manageable section
xmin = 70
xmax = 80
ids = (x>=xmin) & (x<=xmax)
x = x[ids]
y = (ruler_marks - run_mean)[ids]
y -= y.min()
y /= y.max()

dx_dp = np.mean(np.gradient(x))


popt, perr = find_spatial_resolution_from_lineout(x, y, dx_dp, smooth_troughs=0.001,
                                         smooth_peaks=0.001, 
                                         plotter=True)


# 30 - 40:      118 ± 10 %

# 250 - 260:    210 ± 10 %


