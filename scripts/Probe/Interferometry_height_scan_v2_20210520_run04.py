#Interferometry_height_scan_v2.py
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
ne = []
ne_err = []
for shot in range(1,36+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    
    popt,perr = LMI.get_guass_fit_ne(filepath, plotter=False)
    
    shot_num.append(shot)
    ne.append(popt[2])
    ne_err.append(perr[2])

shot_num = np.array(shot_num)
ne = np.array(ne)
ne_err = np.array(ne_err)

#%%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

def nozzle_height(s):
    """from shot sheet
    """
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


log_filedir = ROOT_DATA_FOLDER + '/../' + 'Automation/Outputs/'
log_name = 'gas_jet_y_scan_20210520_run04.txt'
log = np.loadtxt(log_filedir + log_name, dtype=str, skiprows=1)
log_shots = np.array(log[:,1], dtype=int)
log_heights = np.array(log[:,2], dtype=float)

def nozzle_height(s):
    idx = list(log_shots).index(s)
    return log_heights[idx]
    

nozzle_full_height = 23.2
nozzle_in_line = 35.383


xs = [nozzle_full_height + (nozzle_in_line - nozzle_height(s)) for s in shot_num]
xs = np.array(xs)
ys = np.array(ne)
yerrs = np.array(ne_err)

plt.figure()
plt.errorbar(xs, y=ys, yerr=yerrs, marker='x', color='k', capsize=2, ls='')
plt.xlabel('Height above throat [mm]')
plt.ylabel('Plateau $n_e$ [cm$^{-3}$]')

#plt.xscale('log')
#plt.yscale('log')

from scipy.optimize import curve_fit 

def y(x, ne_0): 
    d = 1 # mm throat diameter
    dy_dx = (7.5 - 0.5) / 23.2
    # x is height from nozzle
    return 0.15 * ne_0 * (0.74 * d / (dy_dx * x) )**(2)

def dy_dne_0(x):
    d = 1
    dy_dx = (7.5 - 0.5) / 23.2
    return np.abs( 0.15 * (0.74 * d/ dy_dx )**2 * x**(-2))

# clean-up
xs = xs[np.isfinite(ys)]  
yerrs = yerrs[np.isfinite(ys)]
ys = ys[np.isfinite(ys)]

# error seeems very low here?
popt, pcov = curve_fit(y, xs, ys, sigma=yerrs, absolute_sigma=True, p0=[1.0e22])
perr = pcov[0][0]**(0.5)
dx = np.linspace(np.nanmin(xs), np.nanmax(xs), 100)

y0 = y(dx, popt)
yerr = dy_dne_0(dx) * perr
plt.plot(dx, y0, color=colors[0])
plt.fill_between(dx, y0, y0+yerr, color=colors[0], alpha=0.25)
plt.fill_between(dx, y0, y0-yerr, color=colors[0], alpha=0.25)

print('At nozzle, ne = ', y(nozzle_full_height, *popt))
print('At throat, ne = ', popt[0], ' ± ', 1e2*perr/popt[0], ' %')

ne0 = popt[0]
from scipy.constants import Boltzmann as kB

n_A = ne0 * 1e6 / 2.0 # He
P = (n_A) * kB * (300) # ne im m-3, T in K, to give in backing P in Pa
P_bar = P/1e5
print('Backing pressure estimate: ', P_bar, 'bar')

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