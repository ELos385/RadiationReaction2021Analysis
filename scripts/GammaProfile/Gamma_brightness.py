#!/usr/bin/python3
# Author: Chris Arran
# Date: October 2021
#
# Looks at the brightness of the gamma beam on the profile screen

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

date= '20210620'
run= 'run10'
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
					debugpath='Debug/',debug=False)
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

middle_half = im_max[np.logical_and(im_max<np.percentile(im_max,75), im_max>np.percentile(im_max,25)) ]
stdev_est = np.std(middle_half) / 0.377693
print("Mean and st. dev. of population: %0.2f +- %0.2f" % (np.mean(im_max),np.std(im_max)) )
print("Mean and st. dev. of middle half: %0.2f +- %0.2f" % (np.mean(middle_half),np.std(middle_half)) )
lower_threshold = np.percentile(im_max,25)
upper_threshold = np.mean(middle_half)+4*stdev_est
hits = im_max > upper_threshold
nulls = im_max < lower_threshold
print("Upper / lower thresholds: %0.2f / %0.2f" % (upper_threshold,lower_threshold) )
print("Number of hits / nulls / total: %i / %i / %i" % (np.sum(hits),np.sum(nulls),len(hits)) )

p0a = axs[0,0].loglog(gof,im_max,'.')
p0b = axs[0,0].loglog(gof[hits],im_max[hits],'.')
p0b = axs[0,0].loglog(gof[nulls],im_max[nulls],'.')
p0c = axs[0,0].loglog(gof,spot_mean,'+')
p0d = axs[0,0].loglog(gof[hits],spot_mean[hits],'+')
p0d = axs[0,0].loglog(gof[nulls],spot_mean[nulls],'+')
axs[0,0].set_xlabel('RMS Residual')
axs[0,0].set_ylabel('image max | spot mean')

p1 = axs[0,1].loglog(im_max,spot_mean,'.')
p1b = axs[0,1].loglog(im_max[hits],spot_mean[hits],'.')
p1c = axs[0,1].loglog(im_max[nulls],spot_mean[nulls],'.')
axs[0,1].set_xlabel('image max')
axs[0,1].set_ylabel('spot mean')

valid = np.logical_and(~np.isinf(im_max),im_max > 0)
p2 = axs[1,0].hist(np.log10(im_max[valid]))
p2b = axs[1,0].hist(np.log10(im_max[hits]))
p2c = axs[1,0].hist(np.log10(im_max[np.logical_and(nulls,valid)]))
axs[1,0].set_xlabel('log10(image max)')
axs[1,0].set_ylabel('Count')

valid = np.logical_and(~np.isinf(spot_mean),spot_mean > 0)
p3 = axs[1,1].hist(np.log10(spot_mean[valid]))
p3b = axs[1,1].hist(np.log10(spot_mean[hits]))
p3b = axs[1,1].hist(np.log10(spot_mean[np.logical_and(nulls,valid)]))
axs[1,1].set_xlabel('log10(spot mean)')
axs[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(filename)

