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
from config import HOME
from lib.pipeline import DataPipeline
from modules.Espec.espec_processing import Espec_proc
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile import rad_per_px
from lib.contour_ellipse import contour_ellipse

date= '20210620'
run= 'run09'
filename = 'a0_testcontour_'+date+'_'+run

tForm_filepath = HOME + 'calib/espec1/espec1_transform_20210622_run01_shot001.pkl'
Espec_cal_filepath = HOME + 'calib/espec1/espec1_disp_cal_20210527_run01_shot001.mat'

gamma_profile='GammaProfile'
espec = 'espec1'

# Read in espec data
espec_Proc = Espec_proc(tForm_filepath,Espec_cal_filepath)
espec_pipeline = DataPipeline(espec, 
			espec_Proc.mean_and_std_beam_energy, 
			single_shot_mode=True)
shot_num, espec_data = espec_pipeline.run('%s/%s'%(date, run),parallel='thread')

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10)
a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_vardiff_contour, 
			single_shot_mode=True)
shot_num2, gamma_data = a0_pipeline.run('%s/%s'%(date, run), 
					parallel='thread')

#a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_debug_image, 
#			single_shot_mode=True)
#shot_num2, debug_ims = a0_pipeline.run('%s/%s'%(date, run), 
#					parallel='thread')
#level = np.exp(-0.5)
#vardiff_data = np.zeros(np.shape(shot_num))
#spotSum_data = np.zeros(np.shape(shot_num))
#angle_data = np.zeros(np.shape(shot_num))
#for i,im in enumerate(debug_ims):
#	[major,minor,x0,y0,phi] = contour_ellipse(im,level=level, debug=True, 
#					debugpath='Debug/')
#	angle_data[i] = phi*180/np.pi
#	vardiff = np.abs(major**2-minor**2) / (-2*np.log(level)) 
#	vardiff_data[i] = vardiff*rad_per_px**2
#	spot = im>level*np.max(im)
#	spotSum_data[i] = np.sum(im[spot])

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
a0_data = a0_Est.a0_estimate_av(vardiff,gammai,gammaf)

fig,axs = plt.subplots(nrows=2,ncols=2)

p0 = axs[0].hist(a0_data[~np.isnan(a0_data)])
axs[0].set_xlabel('$a_0$')
axs[0].set_ylabel('Count')

spotSum = gamma_data[:,1]
p1 = axs[1].semilogx(spotSum,a0_data,'.')
axs[1].set_xlabel('spotSum')
axs[1].set_ylabel('$a_0$')

angles = gamma_data[:,2]
p2 = axs[2].hist(angles)
axs[2].set_xlabel('$\phi (^\circ)$')
axs[2].set_ylabel('Count')

p3 = axs[3].plot(gammaf*0.511,a0_data,'.')
axs[3].set_xlabel('$E_f$ (MeV)')
axs[3].set_ylabel('$a_0$')

plt.savefig(filename)

