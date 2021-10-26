#!/usr/bin/python3
# Author: Chris Arran
# Date: October 2021
#
# Identifies collisions as candidates for radiation reaction
# Uses purely the gamma profile brightness in order to avoid any circular reasoning

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from warnings import warn
from config import HOME, ROOT_DATA_FOLDER
from lib.pipeline import DataPipeline
from lib.sql_tools import get_sql_data
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile.GammaProfile import rad_per_px, roi

gamma_profile='GammaProfile'

date= '20210620'
runs= ['run09','run10']
bg_run = 'run04'
filename = 'blinding_test_'+date+'_'+run

null_size = 20
sample_size = 10

gamma_bg_filepath = ROOT_DATA_FOLDER + gamma_profile + '/' + date + '/' + bg_run

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10,bg_path=gamma_bg_filepath,roi=roi)
a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_spot_brightness, 
			single_shot_mode=True)
gsns = brightness = np.array([])
for run in runs:
	shot_num, b = a0_pipeline.run('%s/%s'%(date, run), parallel='thread')
	daterun = ['%s/%s'%(date, run)]
	gsn,datetime = get_sql_data(daterun,shot_num)
	np.append(gsns,gsn)
	np.append(brightness,b)

# Categorise and create subsets

hits = list(np.where(brightness > 1e3)[0])
nulls = list(np.where(brightness < 1e1)[0])
all_shots = range(len(brightness))

A1 = sample(nulls,null_size)
A2 = sample(hits,sample_size)

B1 = np.array(sample(nulls,null_size)).astype(int)
B2 = np.array(sample(nulls,sample_size)).astype(int)

C1 = np.array(sample(all_shots,null_size)).astype(int)
C2 = np.array(sample(all_shots,sample_size)).astype(int)

# Print GSNs to secret files

with open('SetA_' + date + '_' + run + '.txt','w') as f:
	np.savetxt(f,gsns[A1],header='Nulls:',fmt='%i')
	np.savetxt(f,gsns[A2],header='Hits: ',fmt='%i')
with open('SetB_' + date + '_' + run + '.txt','w') as f:
	np.savetxt(f,gsns[B1],header='Nulls:',fmt='%i')
	np.savetxt(f,gsns[B2],header='Hits: ',fmt='%i')
with open('SetC_' + date + '_' + run + '.txt','w') as f:
	np.savetxt(f,gsns[C1],header='Nulls:',fmt='%i')
	np.savetxt(f,gsns[C2],header='Hits: ',fmt='%i')

# Plot
fig,axs = plt.subplots(nrows=2,ncols=2)

valid = np.logical_and(~np.isinf(brightness),brightness > 0)
hrange = np.log10([np.min(brightness[valid]), np.max(brightness[valid])])
p0 = axs[0,0].hist(np.log10(brightness[valid]))
axs[0,0].set_xlabel('log10(pixel brightness)')
axs[0,0].set_ylabel('Population')

p1a = axs[1,0].hist(np.log10(brightness[A1]), range=hrange)
p1b = axs[1,0].hist(np.log10(brightness[A2]), range=hrange)
axs[1,0].set_xlabel('log10(pixel brightness)')
axs[1,0].set_ylabel('Set A')

p1a = axs[0,1].hist(np.log10(brightness[B1]), range=hrange)
p1b = axs[0,1].hist(np.log10(brightness[B2]), range=hrange)
axs[0,1].set_xlabel('log10(pixel brightness)')
axs[0,1].set_ylabel('Set B')

p1a = axs[1,1].hist(np.log10(brightness[C1]), range=hrange)
p1b = axs[1,1].hist(np.log10(brightness[C2]), range=hrange)
axs[1,1].set_xlabel('log10(pixel brightness)')
axs[1,1].set_ylabel('Set C')

plt.tight_layout()
plt.savefig(filename)

