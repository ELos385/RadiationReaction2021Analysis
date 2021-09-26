#Interferometry_height_scan.py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

#date = '20210621'
#run = 'run04'

date = '20210520'
run = 'run04'

shot = 1
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/run99'
LMI = Interferometry(run_name,shot_num=1,diag=diag)

#%%

# can't get this to work?
#LMI_pipeline = DataPipeline(LMI.diag, LMI.get_ne_lineout_from_img, single_shot_mode=True)
#shot_num, ne_data = LMI_pipeline.run('%s/%s'%(date, run))

# use like LivePlotting instead...
shot_num = []
ne_data = []
for shot in range(1,36+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    
    ne = LMI.get_ne_lineout(filepath)
    
    shot_num.append(shot)
    ne_data.append(ne)

shot_num = np.array(shot_num)
ne_data = np.array(ne_data)

#%%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

def nozzle_height(s):
    h = np.nan
    if s>=1 and s<=5: h = 30.383
    if s>=6 and s<=10: h = 26.383
    if s>=11 and s<=15: h = 22.383
    if s>=16 and s<=20: h = 22.383
    if s>=21 and s<=25: h = 18.383
    if s>=21 and s<=25: h = 18.383
    if s>=26 and s<=30: h = 14.383
    if s>=31 and s<=36: h = 10.383
    return h

nozzle_full_height = 23.2
nozzle_in_line = 35.383

xs = []
ys = []
yerrs = []

def find_plateau(y):
    peak = np.nanpercentile(y, 90)
    
    y_good = y[np.isfinite(y)]
    ids_good = (y_good >= peak / 2.0)
    plateau = np.mean(y_good[ids_good])
    
    ids = y >= peak / 2.0
    
    return plateau, ids


for idx in range(len(shot_num)):    
    y = ne_data[idx, 0, :]  
    y_upper = ne_data[idx, 1, :]
    y_lower = ne_data[idx, 2, :]
    
    y0, ids = find_plateau(y) 
    yerr = np.abs(np.nanmean(y_upper[ids] - y_lower[ids]))
    #yerr = find_plateau(y_upper) - find_plateau(y_lower)
    
    
    #plt.plot(x, ne_data[idx, 0, :], label=shot_num[idx])
    #plt.axhline(y=plateau)
    
    
    x = nozzle_full_height + (nozzle_in_line - nozzle_height(shot_num[idx]))
    
    xs.append(x)
    ys.append(y0)
    yerrs.append(yerr)

xs = np.array(xs)
ys = np.array(ys)
yerrs = np.array(yerrs)

plt.figure()
plt.errorbar(xs, y=ys, yerr=yerrs, marker='x', color='k', capsize=2, ls='')
plt.xlabel('Height above throat [mm]')
plt.ylabel('Plateau $n_e$ [cm$^{-3}$]')

#plt.xscale('log')
#plt.yscale('log')

from scipy.optimize import curve_fit 

def y(x, ne_0): 
    d = 1 # mm throat diameter
    # x is height from nozzle
    return 0.15 * ne_0 * (0.74 * d / x)**(2)

def dy_dne_0(x):
    return np.abs( 0.15 * (0.74)**2 * x**(-2))

# clean-up
xs = xs[np.isfinite(ys)]  
yerrs = yerrs[np.isfinite(ys)]
ys = ys[np.isfinite(ys)]

popt, pcov = curve_fit(y, xs, ys, sigma=yerrs, absolute_sigma=True, p0=[4.8e22])
perr = pcov[0][0]**(0.5)
dx = np.linspace(np.nanmin(xs), np.nanmax(xs), 100)

y0 = y(dx, popt)
yerr = dy_dne_0(dx) * perr
plt.plot(dx, y0, color=colors[0])
plt.fill_between(dx, y0, y0+yerr, color=colors[0], alpha=0.25)
plt.fill_between(dx, y0, y0-yerr, color=colors[0], alpha=0.25)

"""

xs2 = np.log10(xs)
ys2 = np.log10(ys)
yerrs2 = np.log10(yerrs)

popt, pcov = np.polyfit(xs2, ys2, deg=1, cov=True)

plt.figure()
#plt.errorbar(xs2, y=ys2, yerr=yerrs2, marker='x', color='k', capsize=2, ls='')
plt.plot(xs2, ys2, marker='x', ls='')
dx = np.linspace(np.nanmin(xs2), np.nanmax(xs2), 100)
plt.plot(dx, popt[0]*dx + popt[1], color=colors[0])

"""