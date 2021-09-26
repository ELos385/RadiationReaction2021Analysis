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
run = 'run11'

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
for shot in range(1,39+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    

    popt,perr = LMI.get_guass_fit_ne(filepath, plotter=False)
    
    # check background offset 
    # print(popt[-1]/popt[-2])
    
    shot_num.append(shot)
    ne.append(popt[2])
    ne_err.append(perr[2])

shot_num = np.array(shot_num)
ne = np.array(ne)
ne_err = np.array(ne_err)




#clean up - shot30 was misfire
clean_up_shots = [16, 38] # bad analysis
for s in clean_up_shots[::-1]:
    idx = list(shot_num).index(s)
    ne = np.delete(ne, idx)
    ne_err = np.delete(ne_err, idx)
    shot_num = np.delete(shot_num, idx)

#%%
# check em

plt.figure()
plt.plot(shot_num, ne, 'x')
plt.xlabel('shot num')
plt.ylabel('ne')

#%%

def pressure(s):
    """from shot sheet
    """
    h = np.nan
    if s>=1 and s<=5: h = 60
    if s>=6 and s<=11: h = 70
    if s>=12 and s<=17: h = 75
    if s>=17 and s<=24: h = 80
    if s>=25 and s<=29: h = 85
    if s>=30: h = 90
    return h

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x = LMI.ne_x_mm

#nozzle_height = 16 mm below laser

xs = [pressure(s) for s in shot_num]
xs = np.array(xs)
ys = np.array(ne)
yerrs = np.array(ne_err)


# offset to make visible
xs_u = list(np.unique(xs))
counts = [0 for i in xs_u]
for x in xs:
    idx = xs_u.index(x)
    counts[idx] += 1

do = 0.25
offsets = [list( do*(np.arange(i) - i/2)) for i in counts]
offsets_final = []
for l in offsets:
    offsets_final += list(l)


plt.figure()
plt.errorbar(xs+offsets_final, y=ys, yerr=yerrs, marker='x', color='k', capsize=2, ls='')
plt.xlabel('Backing Pressure [bar]')
plt.ylabel('$n_e$ [cm$^{-3}$]')

#plt.xscale('log')
#plt.yscale('log')

from scipy.optimize import curve_fit 

def y(x, conv):
    # should just be linear
    return conv * x

def dy_dne_0(x):
    return np.abs( y(x, 1.0))

# clean-up
xs = xs[np.isfinite(ys)]  
yerrs = yerrs[np.isfinite(ys)]
ys = ys[np.isfinite(ys)]

# error seeems very low here?
ecf = 2.0
popt, pcov = curve_fit(y, xs, ys, sigma=ecf * yerrs, absolute_sigma=True, p0=[1.0e13])
perr = pcov[0][0]**(0.5)
dx = np.linspace(np.nanmin(xs), np.nanmax(xs), 100)

y0 = y(dx, popt)
yerr = dy_dne_0(dx) * perr
plt.plot(dx, y0, color=colors[0])
plt.fill_between(dx, y0, y0+yerr, color=colors[0], alpha=0.25)
plt.fill_between(dx, y0, y0-yerr, color=colors[0], alpha=0.25)

conv0 = popt[0]
from scipy.constants import Boltzmann as kB

print('Found conversion: ', conv0, ' ± ', 1e2 * perr/popt[0], ' %')

dP_dne = (1e6 / 2.0) *  kB * (300) / 1e5

height_factor = 0.0005803762924430241
print('Theoretical conversion: ', 1.0/dP_dne * height_factor)



"""
n_A = ne0 * 1e6 / 2.0 # He
P = (n_A) * kB * (300) # ne im m-3, T in K, to give in backing P in Pa
P_bar = P/1e5


"""
