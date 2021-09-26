#Interferometry_height_scan_flows.py
"""
Try using a Gauss fit method to give peak n_e on axis.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.sql_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

#date = '20210621'
#run = 'run04'

date = '20210520'
run = 'run08'

shot = 1
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/' + run
LMI = Interferometry(run_name,shot_num=1,diag=diag)



LMI.fixed_channel_centre = (76.0, 18.5)

#%%

# use like LivePlotting instead...
shot_num = []
ne = []
ne_err = []

clean_up_shots = [1,2]

for shot in range(1,33+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    
    avg,top,bottom = LMI.get_ne_lineout(filepath)
    
    print(LMI.centre)
    
    if shot in clean_up_shots:
        pass
    else:
        ne.append(avg)
        shot_num.append(shot)
        nerr = 0.5 * (np.abs(top - avg) + np.abs(bottom - avg))
        ne_err.append(nerr)
    
shot_num = np.array(shot_num)
ne = np.array(ne)
ne_err = np.array(ne_err)  

#%%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

def blade_z(s):
    """from shot sheet
    """
    h = np.nan
    if s>=1 and s<=7: h = 20.0
    if s>=8 and s<=12: h = 19.0
    if s>=13 and s<=17: h = 18.
    if s>=18 and s<=22: h = 17.0
    if s>=23 and s<=27: h = 16.0
    if s>=28 and s<=38: h = 15.0
    return h

#nozzle_full_height = 23.2
#nozzle_in_line = 19.383 + 16.0

zs = [blade_z(s) for s in shot_num]

plt.figure()
ne_norm = ne

zs_u = list(np.unique(zs))
for s in shot_num:
    idx = list(shot_num).index(s)
    i = ne_norm[idx]
    z = zs[idx]
    z_idx = zs_u.index(z)
    
    plt.plot(x, i, color=colors[z_idx])
    #if z==17 or z==19:
    #    plt.plot(x, i, color=colors[z_idx])

plt.xlabel('Laser axis [mm]')
plt.ylabel('Height above nozzle [mm]')

# difference in peaks between z= 17 and 19 mm are 27.547 - 24.025  = 3.52 mm

# blade IN was y = 13 mm, z = 40 mm at -40 deg - assume this is 4 mm above?
# jet height was 14.383 = 21 mm above nozzle!
# so height between blade and laser is 17 mm
# blade translates 2 mm, but shock translates 3.52 mm
# shock angle increase is 1.52 / 17 = 5.1 deg
# 

#%%
h_us = list(np.unique(heights))
y_us = [[] for i in h_us]
yerr_us = [[] for i in h_us]

for i in range(len(heights)):
    idx = h_us.index(heights[i])
    y_us[idx].append(ne[idx])
    yerr_us[idx].append(ne_err[idx])
    
best_guess_y =  []
best_guess_err = []

for set_idx in range(len(h_us)):
    yz = y_us[set_idx]
    
    num = np.zeros_like(x)
    denom = np.zeros_like(x)
    
    for idx in range(len(yz)):
        y = y_us[set_idx][idx]
        o = yerr_us[set_idx][idx]
        
        num += y/o**2
        denom += 1.0/o**2
    
    bg = num / denom
    err = denom**(-0.5)
    
    best_guess_y.append(bg)
    best_guess_err.append(err)

best_guess_y = np.array(best_guess_y)
best_guess_err = np.array(best_guess_err)

plt.figure()
f = 1e18
best_guess_norm = best_guess_y / f
best_guess_err_norm = (best_guess_err / f) * 6**(0.5)

[plt.plot(x, i+h) for i,h in zip(best_guess_norm, h_us)]
[plt.fill_between(x, y1=h+i-io, y2=h+i+io, alpha=0.25) for i,io,h in zip(best_guess_norm, best_guess_err_norm, h_us)]

[plt.axhline(y=h, color='k', ls='-') for h in heights]


#%%
plt.figure()
[plt.plot(x, i, label='%2.f' % h) for i,h in zip(best_guess_y, h_us)]
[plt.fill_between(x, y1=i-io, y2=i+io, alpha=0.25) for i,io,h in zip(best_guess_y, best_guess_err, h_us)]
plt.legend()

#[plt.axhline(y=h, color='k', ls='-') for h in heights]