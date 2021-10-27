#!/usr/bin/python3
# Author: Chris Arran
# Date: October 2021
#
# Identifies nulls and collisions as candidates for radiation reaction
# Then scrambles samples of nulls and collisions together into blinded data sets 1, 2, 3
# Uses purely the gamma profile brightness in order to avoid any circular reasoning

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from warnings import warn
from config import HOME, ROOT_DATA_FOLDER, BLIND_DATA_FOLDER
from lib.pipeline import DataPipeline
from lib.sql_tools import get_sql_data
from lib.make_blind_dataset import blind_read_and_copy
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile.GammaProfile import rad_per_px, roi

gamma_profile='GammaProfile'

date= '20210620'
#runs= ['run06','run07','run08','run09','run10','run11']
runs = ['run09']
bg_run = 'run12'
filename = 'blinding_test_'+date

#null_size = 100
#sample_size = 100
null_size = 5
sample_size = 5

#gamma_bg_filepath = ROOT_DATA_FOLDER + gamma_profile + '/' + date + '/' + bg_run
gamma_bg_filepath = None
blind_folder = BLIND_DATA_FOLDER + '/' + date + '/'

# Read in GammaProfile data
a0_Est = a0_Estimator(rad_per_px,medfiltwidth=10,bg_path=gamma_bg_filepath,roi=roi)
a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_spot_brightness, 
			single_shot_mode=True)
gsns = brightness = np.array([])
for run in runs:
	shot_num, b = a0_pipeline.run('%s/%s'%(date, run), parallel='thread')
	daterun = ['%s/%s'%(date, run)]
	gsn,datetime = get_sql_data(daterun,shot_num)
	gsns = np.append(gsns,gsn)
	brightness = np.append(brightness,b)
	print("Finished " + run)
	print("Found %i shots, giving %i in total" % (len(shot_num),len(gsns)) )

# Categorise and create subsets

upper = np.percentile(brightness,75)
lower = np.percentile(brightness,25)
middle_half = brightness[ np.logical_and(brightness>lower, brightness<upper) ]
stdev_est = np.std(middle_half) / 0.377693
upper_threshold = np.mean(middle_half) + 4*stdev_est

hits = list(np.where(brightness >= upper_threshold)[0])
nulls = list(np.where(brightness <= lower)[0])
all_shots = range(len(brightness))

print("Sorting hits/nulls by brightness thresholds: >=%0.2f / <=%0.2f" % (upper_threshold,lower) )
print("Number of hits/nulls/total: %i / %i / %i" % (len(hits),len(nulls),len(all_shots)) )
print("Sampling number of hits/nulls: %i / %i" % (sample_size,null_size) )

A1 = sample(nulls,null_size)
A2 = sample(hits,sample_size)

B1 = sample(nulls,null_size)
B2 = sample(nulls,sample_size)

C1 = sample(all_shots,null_size)
C2 = sample(all_shots,sample_size)

# Print GSNs to secret files
in_names = ['SetA','SetB','SetC']
out_names = ['Set1','Set2','Set3']
scrambled = sample(out_names,len(out_names))
header = "Blinded data sets from %s, using runs: %s" % (date,runs)

np.savetxt('blinding_' + date + '.txt', np.transpose([in_names,scrambled]), fmt='%s', header=header)

np.savetxt("%s_nulls_%s.txt" % (scrambled[0],date),gsns[A1],header=header,fmt='%i')
np.savetxt("%s_hits_%s.txt" % (scrambled[0],date),gsns[A2],header=header,fmt='%i')

np.savetxt("%s_nulls_%s.txt" % (scrambled[1],date),gsns[B1],header=header,fmt='%i')
np.savetxt("%s_hits_%s.txt" % (scrambled[1],date),gsns[B2],header=header,fmt='%i')

np.savetxt("%s_nulls_%s.txt" % (scrambled[2],date),gsns[C1],header=header,fmt='%i')
np.savetxt("%s_hits_%s.txt" % (scrambled[2],date),gsns[C2],header=header,fmt='%i')

# Copy files over to blind data
for out_name in out_names:
	filename = "%s_nulls_%s" % (out_name,date)
	blind_read_and_copy(filename+'.txt',blind_folder+filename)
	filename = "%s_hits_%s" % (out_name,date)
	blind_read_and_copy(filename+'.txt',blind_folder+filename)

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

