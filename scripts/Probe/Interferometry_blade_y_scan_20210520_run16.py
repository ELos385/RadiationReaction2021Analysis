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

# He gas, 100 bar backing pressure
# gas jet at y = 14.383, z = 38.82.
# 5 mm below axis is y = 30.383 so starts 21 mm below
# z = 38.83 is south inner edge of jet at south TCC

# nominal blade is at y = 13, z = 20, angle = -40
# can check these with LMS images!

# run 15 blade y scan with automation, blade at 17.7 in z
# run 16 another blade y scan with automation, blade 1 mm further in to 18.5 in z
# run 17 blade z scan at y = 20. 

date = '20210520'
run = 'run16'

shot = 1
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/' + run
LMI = Interferometry(run_name,shot_num=1,diag=diag)
#LMI.fixed_channel_centre = (76.0, 18.0)

#%%

# use like LivePlotting instead...
shot_num = []
ne = []
ne_err = []

clean_up_shots = []

for shot in range(1,30+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    
    avg,top,bottom = LMI.get_ne_lineout(filepath)
    
    #print(LMI.centre)
    
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

x = LMI.ne_x_mm


# inital check
plt.figure()
[plt.plot(x, ne[i]) for i in range(len(ne))]    
#[plt.fill_between(x, ne[i]-ne_err[i], ne[i]+ne_err[i], alpha=0.25) for i in range(len(ne))]  

#%%

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

log_path = ROOT_DATA_FOLDER + '/../Automation/Outputs/blade_y_scan_20210520_run15.txt'
log_data = np.loadtxt(log_path, skiprows=1, dtype=str)
shots = np.array(log_data[:,1], dtype=int)
shots = np.arange(1,30+1) # written wrong into log file
blade_ys = np.array(log_data[:,2], dtype=float)

#%%
h_us = list(np.unique(blade_ys))
y_us = [[] for i in h_us]
yerr_us = [[] for i in h_us]

for i in range(len(blade_ys)):
    idx = h_us.index(blade_ys[i])
    y_us[idx].append(ne[i])
    yerr_us[idx].append(ne_err[i])

def xlim(h):
    """hard code as quicker here
    """
    xl, xu = np.nan, np.nan
    if h==10 or h==12:
        xl,xu = 26, 30
    if h==14 or h==16:
        xl,xu = 27, 38
    if h==18 or h==20:
        xl,xu = 25, 28
        
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
            plt.plot(x, y_us[i][j], color=colors[i], label=blade_h)
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
best_guess_err_norm = (best_guess_err / f) * 6**(0.5)

[plt.plot(x, i+h) for i,h in zip(best_guess_norm, h_us)]
[plt.fill_between(x, y1=h+i-io, y2=h+i+io, alpha=0.25) for i,io,h in zip(best_guess_norm, best_guess_err_norm, h_us)]

[plt.axhline(y=h, color='k', ls='-') for h in blade_ys]


#%%
plt.figure()
[plt.plot(x, i, label='%2.f' % h) for i,h in zip(best_guess_y, h_us)]
[plt.fill_between(x, y1=i-io, y2=i+io, alpha=0.25) for i,io,h in zip(best_guess_y, best_guess_err, h_us)]
plt.legend()

#[plt.axhline(y=h, color='k', ls='-') for h in heights]

#%%
# LMS measurements
# 83 um per pixel

# run15, shot 30
# blade is at (354, 323)
# corresponds to y = 20
# shock at (331, 228) 

# shot 24
# shock at (335, 230)

# shot 19
# shock at (341, 229)

# shot 15
# shock at (350, 229)


# gas jet from run04 shot001
# south OUTER edge is at (394, 293) for y = 30.383 5 mm below axis
# outer edge is 1.5 mm from south inner edge - 7.5 mm from centre

# souht outer edge at 5 mm is (32.7, 24.32) on LMS
# souht inner edge at 5 mm is (31.2, 24.32)
# souht inner edge at 21 mm below is (31.2, 40.32) # increasing as image y increases down page

# USE LMI INSTEAD
# top, south outer edge on LMI is (345, 274)
# so same process gives (27.14, 38.742) for south inner edge of jet on LMI

# on LMI, shot 30 shows
# blade at (302, 337) => (25.07, 27.97) for 20 mm 
# this is the lower shadow so add on 30 pixels in height for top => (25.07, 25.481)
# top channel is at (290, 213) => (24.07, 17.68)
# blade_z = 25.07, gj_z = 27.14, therefore blade 2.07 mm in, 2 mm

# for run 16 blade at (311, 337) => (25.81, 27.97) so blade 2.81 mm in

plt.figure()
f = 1e17
[plt.plot(x, 21.0 + i/f, label='%2.f' % h) for i,h in zip(best_guess_y, h_us)]
[plt.fill_between(x, y1=21.0 + (i-io)/f, y2=21.0 + (i+io)/f, alpha=0.25) for i,io,h in zip(best_guess_y, best_guess_err, h_us)]
plt.legend()

plt.plot( [27.14, np.nanmax(x)], [0.0, 0.0], color='grey') # jet south edge

for idx in range(len(h_us)):
    h = h_us[idx]
    y_tip = (21 - (25.481 - 17.68)) - (20-h)
    
    l = 5.0
    theta = 40.0 * np.pi/180.0
    
    x0,y0 = 25.07, y_tip
    x1,y1 = x0 + l*np.cos(theta), y0 + l*np.sin(theta)
    plt.plot([x0,x1], [y0,y1], color=colors[idx])

#plt.axis('equal')
    
#%%
# simple way by hand
y_tip =  [(21 - (25.481 - 17.68)) - (20-h) for h in h_us]

z_shock = [28.4, 27.7, 26.2, 26.5, 25.6, 25.2]
z_shock = np.array(z_shock)
z_shock = 27.14 - z_shock

plt.figure()
plt.plot(y_tip, z_shock, 'x')
plt.xlabel('Blade height over nozzle [mm]')
plt.ylabel('Shock position from south inner edge [mm]')

#%%
# better way using fwhm for shock pos

# offset to make visible
counts = [0 for i in h_us]
for b in blade_ys:
    idx = h_us.index(b)
    counts[idx] += 1
do = 0.1
offsets = [list( do*(np.arange(i) - i/2)) for i in counts]
offsets_final = []
for l in offsets:
    offsets_final += list(l)
offsets_final = np.array(offsets_final)


y_tip =  [(21 - (25.481 - 17.68)) - (20-h) for h in blade_ys]
y_tip_o = y_tip - offsets_final

z_shock = np.array(shock_pos)
z_shock = 27.14 - z_shock
z_shock_err = np.array(shock_pos_err)

# ignore all y=10 values, looks like blade hadn't moved yet as profile completely inconsistent
i0 = 5
y_tip = y_tip[i0:]
y_tip_o = y_tip_o[i0:]
z_shock = z_shock[i0:]
z_shock_err = z_shock_err[i0:]


plt.figure()
plt.errorbar(y_tip_o, z_shock, yerr=z_shock_err, marker='x', ls='', capsize=2, color='k')
plt.xlabel('Blade height over nozzle [mm]')
plt.ylabel('Shock position from south inner edge [mm]')

y_tip16, y_tip_o16, z_shock16, z_shock_err16 = y_tip, y_tip_o, z_shock, z_shock_err


#%%
# ASSUME YOU HAVE RUN FOR BOTH 15 and 16 HERE
# PLOT BOTH

plt.figure()
plt.errorbar(y_tip_o16, z_shock16, yerr=z_shock_err16, marker='x', ls='', capsize=2, color=colors[0], label='Blade $z=2.1$ mm')
plt.errorbar(y_tip_o15, z_shock15, yerr=z_shock_err15, marker='x', ls='', capsize=2, color=colors[1], label='Blade $z=2.8$ mm')
plt.xlabel('Blade height over nozzle [mm]')
plt.ylabel('Shock $z$ position [mm]')
plt.legend(loc=4)

from scipy.optimize import curve_fit
def func(x,a,b,c):
    return a*x**2 + b*x + c

popt,pcov = curve_fit(func, y_tip16, z_shock16, sigma=z_shock_err16, absolute_sigma=False)
perr = np.diagonal(pcov)**(0.5)
dx = np.linspace(np.min(y_tip16), np.max(y_tip16), 100)
plt.plot(dx, func(dx, *popt), color=colors[0])
#plt.fill_between(dx, func(dx, *popt), func(dx, *(popt+perr)), color=colors[0], alpha=0.25)
#plt.fill_between(dx, func(dx, *popt), func(dx, *(popt-perr)), color=colors[0], alpha=0.25)
print(popt)


popt,pcov = curve_fit(func, y_tip15, z_shock15, sigma=z_shock_err15, absolute_sigma=False)
perr = np.diagonal(pcov)**(0.5)
dx = np.linspace(np.min(y_tip15), np.max(y_tip15), 100)
plt.plot(dx, func(dx, *popt), color=colors[1])
#plt.fill_between(dx, func(dx, *popt), func(dx, *(popt+perr)), color=colors[1], alpha=0.25)
#plt.fill_between(dx, func(dx, *popt), func(dx, *(popt-perr)), color=colors[1], alpha=0.25)
print(popt)