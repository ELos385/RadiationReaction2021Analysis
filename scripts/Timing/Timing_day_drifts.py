#Timing_day_drifts.py

import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Timing.Timing import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from datetime import datetime as dt   

#%%
"""
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
"""
#%%
plt.figure()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

uni_rs = list(np.unique(rs))

colours = np.array([uni_rs.index(r) % len(colors) for r in rs], dtype=int)
colours = np.array(colors)[colours]

plt.figure()
plt.scatter(np.arange(len(ss)), delays, marker='x', color=colours)
plt.xlabel('Shot #'), plt.ylabel('Delay [fs]'), plt.title(date)

# get time of shot        
sql = Read_SQL_shot_summary()
sql_data = sql.get_all()
sql_dt_format = "%Y-%m-%d %H:%M:%S.%f"

runs_sql_format = [date + '/' + r for r in rs]
timestamps = []
for idx, r in enumerate(runs_sql_format):
    ts = np.nan
    
    try:
        ids = (sql_data['run']==r) & (sql_data['shot_or_burst']==str(ss[idx]))
        timestamp = np.array(sql_data[ids]['timestamp'])[0]
        ts = dt.strptime(timestamp, sql_dt_format)
    except(IndexError):
        print('Failed for %s %s' % (r, ss[idx]))
    
    timestamps.append(ts)

plt.figure()
plt.scatter(timestamps, delays, marker='x', color=colours)
plt.xlabel('Shot #'), plt.ylabel('Delay [fs]'), plt.title(date)