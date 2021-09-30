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
run = 'run17'

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

clean_up_shots = []

for shot in range(1,36+1):
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

x = LMI.ne_x_mm

log_path = ROOT_DATA_FOLDER + '/../Automation/Outputs/blade_z_scan_20210520_run17.txt'
log_data = np.loadtxt(log_path, skiprows=1, dtype=str)
#log_shots = list(np.array(log_data[:,1], dtype=int))
log_shots = list(np.arange(1,36+1))

# this looks wrong!
#log_blade_zs = np.array(log_data[:,2], dtype=float)
#log_blade_zs = [20]*5 + [19.5]*5 + [19]*5 + [18.5]*5 + [18]*5 + [17.5]

# check rouhgly from images
LMI_shots = [5, 10, 15, 20, 25, 26, 31, 36]
LMI_x_vals= [329, 321, 317, 309, 303, 298, 325, 322]
LMI_x_vals = np.array(LMI_x_vals)
LMI_x_vals = 0.083 * LMI_x_vals
# (27.14, 38.742) for south inner edge of jet on LMI
LMI_x_vals = 27.14 - LMI_x_vals

blade_zs = [LMI_x_vals[0]]*5 + [LMI_x_vals[1]]*5 + [LMI_x_vals[2]]*5  + [LMI_x_vals[3]]*5  + [LMI_x_vals[4]]*5  + [LMI_x_vals[5]]*1 + [LMI_x_vals[6]]*5 + [LMI_x_vals[7]]*5   
blade_zs = np.array(blade_zs)
#blade_zs = [log_blade_zs[log_shots.index(s)] for s in shot_num]
#%%
h_us = list(np.unique(blade_zs))
y_us = [[] for i in h_us]
yerr_us = [[] for i in h_us]


for i in range(len(blade_zs)):
    idx = h_us.index(blade_zs[i])
    y_us[idx].append(ne[i])
    yerr_us[idx].append(ne_err[i])

def xlim(h):
    """hard code as quicker here
    """
    xl, xu = np.nan, np.nan
    if h==18:
        xl,xu = 23, 25
    if h==18.5:
        xl,xu = 25.5, 28
    if h==19:
        xl,xu = 26.0, 28
    if h==19.5:
        xl,xu = 27.6, 31
    if h==19.75:
        xl,xu = 27.0, 29
    if h==20:
        xl,xu = 27.5, 31 
    return xl, xu


def xlim(h):
    """hard code as quicker here
    """
    xl, xu = np.nan, np.nan
    if h==h_us[7]:
        xl,xu = 23, 25
    if h==h_us[6]:
        xl,xu = 24, 26
    if h==h_us[5]:
        xl,xu = 25, 27
    if h==h_us[4]:
        xl,xu = 26, 28
    if h==h_us[3]:
        xl,xu = 27.5, 30
    if h==h_us[2]:
        xl,xu = 27.0, 31
    if h==h_us[1]:
        xl,xu = 27.0, 29
    if h==h_us[0]:
        xl,xu = 27.5, 31 
    return xl, xu

shock_pos = []
shock_pos_err = []
ecf = 0.5
# plot all - colour coded
plt.figure()
for i in range(len(h_us)):
    i = i
    blade_h = h_us[i]
    y_set = y_us[i]
    for j in range(len(y_set)):
        if j == 0:
            plt.plot(x, y_us[i][j], color=colors[i], label='%.3f' % blade_h)
        else:
            y0 = y_us[i][j]
            ye0 = yerr_us[i][j] * ecf
            plt.plot(x, y0, color=colors[i])
            plt.fill_between(x, y0-ye0, y0+ye0, color=colors[i], alpha=0.25)
        
        xl, xu = xlim(blade_h)
        ids = (x>=xl) & (x<=xu)
        y = y_us[i][j][ids]
        
        #idx = np.nanargmin(test) + np.arange(x.size)[ids][0]
        idx = np.nanargmax(y) + np.arange(x.size)[ids][0]
        x0 = x[idx]
        plt.axvline(x=x0, ls='-', color=colors[i])
        
        y_low = y_us[i][j] - (yerr_us[i][j] * ecf)
        y_high = y_us[i][j] + (yerr_us[i][j] * ecf)
        y_low_peak_val = y_low[idx]
        y_high_over_low_peak_ids = y_high >= y_low_peak_val
        x_range = x[y_high_over_low_peak_ids]
        xl, xr = x_range[0], x_range[-1]
        xo_avg = 0.5 * ((x0-xl) + (xr-x0))
        
        #print(x0, x0-xl, xr-x0, xo_avg)      
        shock_pos.append(x0)
        shock_pos_err.append(xo_avg)
        

plt.legend()

#%%    
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
best_guess_err_norm = (best_guess_err / f) 

[plt.plot(x, i, label=h) for i,h in zip(best_guess_norm, h_us)]
[plt.fill_between(x, y1=i-io, y2=i+io, alpha=0.25) for i,io,h in zip(best_guess_norm, best_guess_err_norm, h_us)]
plt.legend()

#[plt.axhline(y=h, color='k', ls='-') for h in blade_zs]

#%%
# simple way by hand
# SUPERSEDED
"""
z_tip =  h_us[:]

z_shock = [24.75, 26.45, 27.55, 24.52, 28.55, 28.80]
z_shock = 27.14 - np.array(z_shock)
z_shock = 27.14 - z_shock

plt.figure()
plt.plot(z_tip, z_shock, 'x')
plt.xlabel('Blade position from south inner edge [mm]')
plt.ylabel('Shock position from south inner edge [mm]')
"""
#%%
# better way using fwhm for shock pos

# offset to make visible
counts = [0 for i in h_us]
for b in blade_zs:
    idx = h_us.index(b)
    counts[idx] += 1
do = 0.025
offsets = [list( do*(np.arange(i) - i/2)) for i in counts]
offsets_final = []
for l in offsets:
    offsets_final += list(l)
offsets_final = np.array(offsets_final)


y_tip =  blade_zs
y_tip_o = y_tip - offsets_final

z_shock = np.array(shock_pos)
z_shock = 27.14 - z_shock
z_shock_err = np.array(shock_pos_err)

# ignore all y=10 values, looks like blade hadn't moved yet as profile completely inconsistent
i0 = 0
y_tip = y_tip[i0:]
y_tip_o = y_tip_o[i0:]
z_shock = z_shock[i0:]
z_shock_err = z_shock_err[i0:]

#ecf = 0.5
plt.figure()
plt.errorbar(y_tip_o, z_shock, yerr=z_shock_err*ecf, marker='x', ls='', capsize=2, color='k')
plt.xlabel('Blade $z$ [mm]')
plt.ylabel('Shock $z$ [mm]') #boht from gj south inner edge


# fits
valid = [h_us[0]] + h_us[3:]
ids = np.array([i for i,j in enumerate(y_tip) if j in valid])
xd = y_tip[ids]
yd = z_shock[ids]
popt = np.polyfit(xd, yd, deg=1)
z = np.poly1d(popt)
x_dummy = np.linspace(np.min(xd), np.max(xd), 100)
plt.plot(x_dummy, z(x_dummy))

valid = h_us[0:3]
ids = np.array([i for i,j in enumerate(y_tip) if j in valid])
xd = y_tip[ids]
yd = z_shock[ids]
popt = np.polyfit(xd, yd, deg=1)
z = np.poly1d(popt)
x_dummy = np.linspace(np.min(xd), np.max(xd), 100)
plt.plot(x_dummy, z(x_dummy))

