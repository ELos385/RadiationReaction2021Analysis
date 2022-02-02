#!/usr/bin/python3
# Author: Chris Arran
# Date: October 2021
#
# Identifies nulls and collisions as candidates for radiation reaction
# Then looks at the differences in a0 estimates, goodness of fit, and angle of ellipse

import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from config import HOME, ROOT_DATA_FOLDER, BLIND_DATA_FOLDER
from lib.pipeline import DataPipeline
from modules.Espec.espec_processing import Espec_proc
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile.GammaProfile import rad_per_px, roi

espec = 'espec1'
gamma_profile='GammaProfile'

date= '20210620'
runs= ['run05','run06','run07','run08','run09','run10','run11']
#runs = ['run05']
bg_run = 'run12'
filename = 'GammaProfile_'+date

tForm_filepath = HOME + 'calib/espec1/espec1_transform_20210622_run01_shot001.pkl'
Espec_cal_filepath = HOME + 'calib/espec1/espec1_disp_cal_20210527_run01_shot001.mat'
gamma_bg_filepath = ROOT_DATA_FOLDER + gamma_profile + '/' + date + '/' + bg_run

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10,bg_path=gamma_bg_filepath,roi=roi)
a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_spot_brightness, 
			single_shot_mode=True)
a0_pipeline2 = DataPipeline(gamma_profile,a0_Est.get_vardiff_contour, 
			single_shot_mode=True)

espec_Proc = Espec_proc(tForm_filepath,Espec_cal_filepath)
espec_pipeline = DataPipeline(espec, 
			espec_Proc.mean_and_std_beam_energy, 
			single_shot_mode=True)

brightness = np.array([])
vdiffs = gammas = phis = gofs = np.array([])
for run in runs:
	shot_num, b = a0_pipeline.run('%s/%s'%(date, run), parallel='thread')
	shot_num2, a0_data = a0_pipeline2.run('%s/%s'%(date, run), parallel='thread')
	shot_num3, espec_data = espec_pipeline.run('%s/%s'%(date, run),parallel='thread')

	brightness = np.append(brightness,b)
	vdiffs = np.append(vdiffs,a0_data[:,0])
	gammas = np.append(gammas,espec_data[:,0]/0.511)
	phis = np.append(phis,a0_data[:,2])
	gofs = np.append(gofs,a0_data[:,3])

	print("Finished " + run)
	print("Found %i shots, giving %i in total" % (len(shot_num),len(brightness)) )

# Categorise and create subsets

upper = np.percentile(brightness,75)
lower = np.percentile(brightness,25)
middle_half = brightness[ np.logical_and(brightness>lower, brightness<upper) ]
stdev_est = np.std(middle_half) / 0.377693

nulls = list(np.where(brightness <= lower)[0])
all_shots = range(len(brightness))

gammai = np.median(gammas[nulls])	# Use median to avoid tails
a0s = a0_Est.a0_estimate_av(vdiffs,gammai,gammas)

thresholds = np.linspace(0,10,11)
up_thresholds = np.zeros_like(thresholds)
N_hits = np.zeros_like(thresholds)  
mean_a0s = np.zeros_like(thresholds)
std_a0s = np.zeros_like(thresholds)
mean_phis = np.zeros_like(thresholds)
std_phis = np.zeros_like(thresholds)

for i,threshold in enumerate(thresholds):
	upper_threshold = np.mean(middle_half) + threshold*stdev_est
	hits = list(np.where(brightness >= upper_threshold)[0])

	up_thresholds[i] = upper_threshold
	N_hits[i] = len(hits)

	print("Sorting hits/nulls by brightness thresholds: >=%0.2f / <=%0.2f" % (upper_threshold,lower) )
	print("Number of hits/nulls/total: %i / %i / %i" % (len(hits),len(nulls),len(all_shots)) )

	# Plot
	fig,axs = plt.subplots(nrows=2,ncols=2)

	valid = np.logical_and(~np.isinf(brightness),brightness > 0)
	hrange = np.log10([np.min(brightness[valid]), np.max(brightness[valid])])
	p0 = axs[0,0].loglog(brightness[valid],a0s[valid],'.',ms=2)
	p0a = axs[0,0].loglog(brightness[nulls],a0s[nulls],'.',ms=2)
	p0b = axs[0,0].loglog(brightness[hits],a0s[hits],'.',ms=2)
	axs[0,0].set_xlabel('pixel brightness')
	axs[0,0].set_ylabel('$a_0$')
	axs[0,0].set_title('Threshold $>\mu+%0.1f\sigma$' % threshold)

	p1 = axs[0,1].loglog(gofs[valid],a0s[valid],'.',ms=2)
	p1a = axs[0,1].loglog(gofs[nulls],a0s[nulls],'.',ms=2)
	p1b = axs[0,1].loglog(gofs[hits],a0s[hits],'.',ms=2)
	axs[0,1].set_xlabel('RMS Residual')
	axs[0,1].set_ylabel('$a_0$')

	p2 = axs[1,0].hist(a0s[valid],range=(0,50))
	p2a = axs[1,0].hist(a0s[nulls],range=(0,50))
	p2b = axs[1,0].hist(a0s[hits])
	axs[1,0].set_xlabel('$a_0$')
	axs[1,0].set_ylabel('Count')
	axs[1,0].set_title( '$a_0 = %0.1f \pm %0.1f$' % (np.mean(a0s[hits]),np.std(a0s[hits])) )

	p3 = axs[1,1].hist(phis[valid])
	p3a = axs[1,1].hist(phis[nulls])
	p3b = axs[1,1].hist(phis[hits])
	axs[1,1].set_xlabel('$\phi (^\circ)$')
	axs[1,1].set_ylabel('Count')
	axs[1,1].set_title( '$\phi = %0.1f \pm %0.1f$' % (np.mean(phis[hits]),np.std(phis[hits])) )

	plt.tight_layout()
	plt.savefig(filename+'_%isigma' % threshold)

	nulls2 = np.intersect1d(nulls,valid)
	print('')
	print('All a0: ',np.mean(a0s[valid]),'+-',np.std(a0s[valid]))
	print('Nulls a0: ',np.mean(a0s[nulls2]),'+-',np.std(a0s[nulls2]))
	print('Hits a0: ',np.mean(a0s[hits]),'+-',np.std(a0s[hits]))
	print('')
	print('All phi: ',np.mean(phis),'+-',np.std(phis))
	print('Nulls phi: ',np.mean(phis[nulls]),'+-',np.std(phis[nulls]))
	print('Hits phi: ',np.mean(phis[hits]),'+-',np.std(phis[hits]))
	print('')

	mean_a0s[i] = np.mean(a0s[hits])
	std_a0s[i] = np.std(a0s[hits])
	mean_phis[i] = np.mean(phis[hits])
	std_phis[i] = np.std(phis[hits])

fig,axs = plt.subplots(nrows=2,ncols=2)

p0 = axs[0,0].plot(thresholds,up_thresholds,'o')
axs[0,0].set_ylabel('Brightness threshold')
axs[0,0].set_xlim(0,10)

p1 = axs[0,1].plot(thresholds,N_hits/len(all_shots),'o')
axs[0,1].set_ylabel('Fraction of hits')
axs[0,1].set_xlim(0,10)

p2 = axs[1,0].errorbar(thresholds,mean_a0s,yerr=std_a0s,marker='o',capsize=4)
axs[1,0].set_ylabel('$a_0$')
axs[1,0].set_xlim(0,10)

p3 = axs[1,1].errorbar(thresholds,mean_phis,yerr=std_phis,marker='o',capsize=4)
axs[1,1].set_ylabel('$\phi$')
axs[1,1].set_xlim(0,10)

axs[1,0].set_xlabel('Threshold/$\sigma$')
axs[1,1].set_xlabel('Threshold/$\sigma$')

plt.tight_layout()
plt.savefig(filename+'_thresholds')


