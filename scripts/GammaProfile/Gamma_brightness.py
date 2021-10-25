#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile screen
# Added the espec to pull out gamma_i and gamma_f
# Added plotting routine to make figure

import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from config import HOME, ROOT_DATA_FOLDER
from lib.pipeline import DataPipeline
from modules.Espec.espec_processing import Espec_proc
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile.GammaProfile import rad_per_px, roi
from lib.contour_ellipse import contour_ellipse, ellipse_mask

gamma_profile='GammaProfile'

date= '20210604'
run= 'run23'
bg_run = 'run04'
filename = 'brightness_test_'+date+'_'+run

gamma_bg_filepath = ROOT_DATA_FOLDER + gamma_profile + '/' + date + '/' + bg_run

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10,bg_path=gamma_bg_filepath,roi=roi)
#a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_vardiff_contour, 
#			single_shot_mode=True)
#shot_num2, gamma_data = a0_pipeline.run('%s/%s'%(date, run), 
#					parallel='thread')

a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_debug_image, 
			single_shot_mode=True)
shot_num, debug_ims = a0_pipeline.run('%s/%s'%(date, run), 
					parallel='thread')
level = 0.5
gamma_data = np.zeros([np.size(shot_num),4])
for i,im in enumerate(debug_ims):
	[major,minor,x0,y0,phi,gof] = contour_ellipse(im,level=level, 
					debugpath='Debug/')
	
	gamma_data[i,0] = np.max(im)
	mask = ellipse_mask(np.shape(im), [major,minor,x0,y0,phi,gof])
	gamma_data[i,1] = np.mean(im*mask)
	gamma_data[i,2] = phi*180/np.pi
	gamma_data[i,3] = gof

# Plot
fig,axs = plt.subplots(nrows=2,ncols=2)

im_max = gamma_data[:,0]
spot_mean = gamma_data[:,1]
gof = gamma_data[:,3]
subset = gof<1e6

p0a = axs[0,0].loglog(gof,im_max,'.')
p0b = axs[0,0].loglog(gof[subset],im_max[subset],'.')
p0c = axs[0,0].loglog(gof,spot_mean,'+')
p0d = axs[0,0].loglog(gof[subset],spot_mean[subset],'+')
axs[0,0].set_xlabel('RMS Residual')
axs[0,0].set_ylabel('image max | spot mean')

p1 = axs[0,1].loglog(im_max,spot_mean,'.')
p1b = axs[0,1].loglog(im_max[subset],spot_mean[subset],'.')
axs[0,1].set_xlabel('image max')
axs[0,1].set_ylabel('spot mean')

p2 = axs[1,0].hist(np.log10(im_max))
axs[1,0].set_xlabel('log10(image max)')
axs[1,0].set_ylabel('Count')

valid = np.logical_and(~np.isinf(spot_mean),spot_mean > 0)
p3 = axs[1,1].hist(np.log10(spot_mean[valid]))
axs[1,1].set_xlabel('log10(spot mean)')
axs[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(filename)

