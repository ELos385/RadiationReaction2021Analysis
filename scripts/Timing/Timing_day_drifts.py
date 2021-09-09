#Timing_day_drifts.py

import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from lib.sql_tools import *
from modules.Timing.Timing import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

#%%

delay = Timing()

#%%

# Grab entire data of diagnostic from given DAY to see long term drifts
diag_timing='LA3timing'

date = '20210618'

runs = get_dirs(diag_timing, date)

rs = []
ss = []
delays = []

for run in runs:
    delay_pipeline = DataPipeline(diag_timing, delay.get_delay_from_img, single_shot_mode=True)
    # single shot mode=True fails if a run is actually called Burst (for whatever reason)
    shot_num_delay, delay_data = delay_pipeline.run('%s/%s'%(date, run))
    delay_data = list(delay_data)
    
    [rs.append(run) for i in delay_data]
    ss += shot_num_delay
    delays += delay_data

rs = np.array(rs)
ss = np.array(ss)
delays = np.array(delays)


runs_sql_format = [date + '/' + r for r in rs]
gsns, timestamps = get_sql_data(runs_sql_format, ss)

#%%

# reorganise following gsn
all_data = np.array([gsns, timestamps, rs, ss, delays]).T
all_data = all_data[all_data[:, 0].argsort()]
gsns, timestamps, rs, ss, delays = all_data.T

# use colours to identify unique runs
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
uni_rs = list(np.unique(rs))
colours_idx = np.array([uni_rs.index(r) % len(colors) for r in rs], dtype=int)
colours = np.array(colors)[colours_idx]

# plot against gsn
plt.figure()
plt.scatter(gsns, delays, marker='x', color=colours)
plt.xlabel('GSN'), plt.ylabel('Delay [fs]'), plt.title(date)
plt.grid()

# fit trend
N_run_mean = 5
run_mean = np.convolve(delays, np.ones(N_run_mean)/N_run_mean, mode='same')
delta = np.abs(delays - run_mean)
percentile_threshold = 99 # %
ids = ~(delta > np.percentile(delta, percentile_threshold))
N_run_mean = 5
run_mean = np.convolve(delays[ids], np.ones(N_run_mean)/N_run_mean, mode='same')
plt.plot(gsns[ids], run_mean, 'k')



# plot against time to account for drifts over dinner etc.
plt.figure()
plt.scatter(timestamps, delays, marker='x', color=colours)
plt.xlabel('Timestamp'), plt.ylabel('Delay [fs]'), plt.title(date)
plt.xlim((timestamps.min(), timestamps.max()))
plt.grid()