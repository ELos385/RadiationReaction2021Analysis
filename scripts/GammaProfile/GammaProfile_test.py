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
from calib.GammaProfile import rad_per_px
from lib.contour_ellipse import contour_ellipse

gamma_profile='GammaProfile'
espec = 'espec1'

date= '20210620'
run= 'run09'
bg_run = 'run04'
filename = 'a0_testcontour_'+date+'_'+run

tForm_filepath = HOME + 'calib/espec1/espec1_transform_20210622_run01_shot001.pkl'
Espec_cal_filepath = HOME + 'calib/espec1/espec1_disp_cal_20210527_run01_shot001.mat'
gamma_bg_filepath = ROOT_DATA_FOLDER + gamma_profile + '/' + date + '/' + bg_run

# Read in espec data
espec_Proc = Espec_proc(tForm_filepath,Espec_cal_filepath)
espec_pipeline = DataPipeline(espec, 
			espec_Proc.mean_and_std_beam_energy, 
			single_shot_mode=True)
shot_num, espec_data = espec_pipeline.run('%s/%s'%(date, run),parallel='thread')

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10,bg_path=gamma_bg_filepath)
#a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_vardiff_contour, 
#			single_shot_mode=True)
#shot_num2, gamma_data = a0_pipeline.run('%s/%s'%(date, run), 
#					parallel='thread')

a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_debug_image, 
			single_shot_mode=True)
shot_num2, debug_ims = a0_pipeline.run('%s/%s'%(date, run), 
					parallel='thread')
level = 0.5
gamma_data = np.zeros([np.size(shot_num),4])
for i,im in enumerate(debug_ims):
	[major,minor,x0,y0,phi,gof] = contour_ellipse(im,level=level, debug=True, 
					debugpath='Debug/')
	vardiff = np.abs(major**2-minor**2) / (-2*np.log(level)) 
	gamma_data[i,0] = vardiff*rad_per_px**2
	spot = im>level*np.max(im)
	gamma_data[i,1] = np.sum(im[spot])
	gamma_data[i,2] = phi*180/np.pi
	gamma_data[i,3] = gof

# Raise a warning if shot numbers don't match
if shot_num2 != shot_num:
	warn(gamma_profile+" and "+espec+" folders contain different shots. Trying to reconcile.",RuntimeWarning)
	print(shot_num)
	print(shot_num2)
	print("->")
	shot_num,is1,is2 = np.intersect1d(shot_num, shot_num2, 
					return_indices=True)
	espec_data = espec_data[is1]
	gamma_data = gamma_data[is2]
	print(shot_num)

# Estimate a0 and plot
gammaf = espec_data[:,0]/0.511	# 90th percentile energies
gammai = np.median(gammaf)	# Assume most shots are misses
vardiff = gamma_data[:,0]
a0 = a0_Est.a0_estimate_av(vardiff,gammai,gammaf)

fig,axs = plt.subplots(nrows=2,ncols=2)

gof = gamma_data[:,3]
subset = np.logical_and(gof<np.exp(-2),~np.isnan(a0))

gof = gamma_data[:,3]
ngof,bins,p2 = axs[0,0].hist(np.log10(gof))
p2b = axs[0,0].hist(np.log10(gof[subset]),bins=bins)
axs[0,0].set_xlabel('RMS Residual (log10)')
axs[0,0].set_ylabel('Count')

spotMean = gamma_data[:,1]
p1 = axs[0,1].loglog(gof,spotMean,'.')
p1b = axs[0,1].loglog(gof[subset],spotMean[subset],'.')
axs[0,1].set_xlabel('RMS Residual')
axs[0,1].set_ylabel('spotMean')

p3 = axs[1,0].loglog(gof,a0,'.')
p3b = axs[1,0].loglog(gof[subset],a0[subset],'.')
axs[1,0].set_xlabel('RMS Residual')
axs[1,0].set_ylabel('$a_0$')

p0b = axs[1,1].hist(a0[subset])
axs[1,1].set_xlabel('$a_0$')
axs[1,1].set_ylabel('Count')


plt.tight_layout()
plt.savefig(filename)

