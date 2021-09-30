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
from lib.sql_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

#date = '20210621'
#run = 'run04'

date = '20210609'
run = 'run06'

shot = 1
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/' + run
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
    
    print(LMI.centre)
    
    shot_num.append(shot)
    ne.append(popt[2])
    ne_err.append(perr[2])

shot_num = np.array(shot_num)
ne = np.array(ne)
ne_err = np.array(ne_err)

#clean up - shot30 was misfire
clean_up_shots = [1, 30]
for s in clean_up_shots[::-1]:
    idx = list(shot_num).index(s)
    ne = np.delete(ne, idx)
    ne_err = np.delete(ne_err, idx)
    shot_num = np.delete(shot_num, idx)




#%%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

def nozzle_height(s):
    """from shot sheet
    """
    h = np.nan
    if s>=1 and s<=6: h = 19.383
    if s>=7 and s<=12: h = 21.383
    if s>=13 and s<=18: h = 23.383
    if s>=19 and s<=25: h = 25.383
    if s>=26 and s<=32: h = 27.383
    if s>=33 and s<=38: h = 29.383
    if s>=39 and s<=53: h = 31.383
    return h

nozzle_full_height = 23.2
nozzle_in_line = 19.383 + 16.0
plt.figure()
plt.plot(shot_num, ne, 'x')
plt.xlabel('shot num')
plt.ylabel('ne')

"""
set_1 = [1,2,3, 8, 12, 13, 23, 25, 26, 27, 28, 24]
set_2 = [4,5,6,7, 9,10,11, 14,15,16,17,18,19,20,21,22, 28, 31, 32, 33, 35, 36] # 24

for s in set_2[::-1]:    
    idx = list(shot_num).index(s)
    ne = np.delete(ne, idx)
    ne_err = np.delete(ne_err, idx)
    shot_num = np.delete(shot_num, idx)
"""
xs = [nozzle_full_height + (nozzle_in_line - nozzle_height(s)) for s in shot_num]
xs = np.array(xs)
ys = np.array(ne)
yerrs = np.array(ne_err)

plt.figure()
plt.errorbar(xs, y=ys, yerr=yerrs, marker='x', color='k', capsize=2, ls='')
plt.xlabel('Height above throat [mm]')
plt.ylabel('$n_e$ [cm$^{-3}$]')

#plt.xscale('log')
#plt.yscale('log')

from scipy.optimize import curve_fit 



def y(x, ne_0):
    # from Chen et al., Applied Physics Letters, (2013)
    d = 1
    dy_dx = (7.5-0.5)/23.2 # tan (alpha)
    gamma = 5/3 # He
    
    t1 = (2.0/(gamma + 1))**(1/(gamma-1))
    t2 = (gamma / (gamma + 1))**(0.5)
    t3 = (gamma / (gamma - 1))**(0.5)
    t4 = d / (2.0 * dy_dx)
    
    return ne_0 * ((t1 * t2) / t3) * t4**2 * x**-2


def dy_dne_0(x):
    return np.abs( y(x, 1.0))

# clean-up
xs = xs[np.isfinite(ys)]  
yerrs = yerrs[np.isfinite(ys)]
ys = ys[np.isfinite(ys)]

# error seeems very low here?
ecf = 15.0 # error correction factor?
popt, pcov = curve_fit(y, xs, ys, sigma=ecf*yerrs, absolute_sigma=True, p0=[1.0e22])
#popt, pcov = curve_fit(y, xs, ys, p0=[1.0e22])
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
print('Backing pressure supposedly 90 bar He-N 2%')

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


ss = [str(s) for s in shot_num]
ss = ['Shot'+'0'*(3 - len(s)) + s for s in ss]

drs = [date + run + s for s in ss]

gsns = [get_gsn(i) for i in drs]

timestamps = np.array([get_timestamp(i) for i in drs])
print(np.diff(timestamps))
# all rouhgly 20 s apart!
"""

#%%
plt.figure()

xs_u = list(np.unique(xs))
counts = [0 for i in xs_u]
for x in xs:
    idx = xs_u.index(x)
    counts[idx] += 1

do = 0.1
offsets = [list( do*(np.arange(i) - i/2)) for i in counts]
offsets_final = []
for l in offsets:
    offsets_final += list(l)


plt.errorbar(xs - nozzle_full_height + offsets_final, y=ys, yerr=yerrs, marker='x', color='k', capsize=2, ls='')
plt.xlabel('Height above nozzle [mm]')
plt.ylabel('$n_e$ [cm$^{-3}$]')
plt.plot(dx- nozzle_full_height, y0, color=colors[0])
plt.fill_between(dx- nozzle_full_height, y0, y0+yerr, color=colors[0], alpha=0.25)
plt.fill_between(dx- nozzle_full_height, y0, y0-yerr, color=colors[0], alpha=0.25)





