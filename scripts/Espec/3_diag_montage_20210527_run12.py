"""Quick inspect a run of shots to see any correlations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

# start with an example shot - same workflow as used in LivePlotting

diag = 'espec1'
charge=-1.0
#date='20210618'
#run= 'run18'

date='20210527'
run= 'run12'

#### replace DATA_FOLDER ####
run_name = date+'/run99'
spec = Espec(run_name,shot_num=1,img_bkg=None,diag=diag)

fp_head = ROOT_DATA_FOLDER + '/' 
fp_tail = '/' + date + '/' + run +'/' + 'Shot'

#%%

# espec settings
E = spec.x_MeV
y = spec.y_mm
dy = np.gradient(y)[0] #uniform
y -= np.mean(y)
y_cutoff = 10
EX, YX = np.meshgrid(E, y)
img_mask = np.abs(YX) < y_cutoff

# f40 settings
f40_mid_y, f40_mid_x = 387, 364
f40_hw = 30

# probe settings
p_mid_y, p_mid_x = 236, 531
p_hw = 30


specs = []
leakage_spots = []
shocks = []

for i in range(157):
    idx = i + 1
    print(i)
    s = str(idx)
    s = '0'*(3-len(s)) + s
    
    # default output if failed
    lineout = np.full_like(E, fill_value=np.nan)
    f40_img = np.full((2*f40_hw, 2*f40_hw), fill_value=np.nan)
    probe_img = np.full((2*p_hw, 2*p_hw), fill_value=np.nan)
    
    try:    
        fn = fp_head + 'espec1' + fp_tail + s + '.tif'
        espec_img = imread(fn)
        espec_img = spec.eSpec_proc.espec_data2screen(espec_img) # pC / mm MeV
        lineout = np.sum(espec_img * img_mask, axis=0) * dy
    except(FileNotFoundError):
        pass

    try:
        fn = fp_head + 'F40Leakage' + fp_tail + s + '.tiff'
        f40_img = imread(fn)
        f40_img = f40_img[f40_mid_y-f40_hw : f40_mid_y + f40_hw, f40_mid_x-f40_hw : f40_mid_x + f40_hw]
    except(FileNotFoundError):
        pass
    
    try:
        fn = fp_head + 'LMS' + fp_tail + s + '.tiff'
        probe_img = imread(fn)
        probe_img = probe_img[p_mid_y-p_hw : p_mid_y + p_hw, p_mid_x-p_hw : p_mid_x + p_hw]
    except(FileNotFoundError):
        pass

    
    specs.append(lineout)    
    leakage_spots.append(f40_img)
    shocks.append(probe_img)
    
#%%
# failed attempt at a montage
"""
import matplotlib.gridspec as gridspec

# make a montage-y plot
N = len(specs)
#fig, axes = plt.subplots(3, N)

fig = plt.figure(figsize=(16,1.75))

for i in range(N):
    f3_ax1 = plt.subplot2grid((5, N), (0, i))
    f3_ax2 = plt.subplot2grid((5, N), (1, i))
    f3_ax3 = plt.subplot2grid((5, N), (2, i), rowspan=3)
    
    #f3_ax1 = fig3.add_subplot(gs[0, i])
    #f3_ax2 = fig3.add_subplot(gs[1, i])
    #f3_ax3 = fig3.add_subplot(gs[2:, i])
    
    f3_ax1.imshow(shocks[i])
    f3_ax2.imshow(leakage_spots[i])
    
    E_min, E_max = 320, 1500
    ids = (E>=E_min) & (E<=E_max)
    f3_ax3.plot(specs[i][ids], E[ids], color='k')
    
    f3_ax1.set_xticks([]), f3_ax1.set_yticks([])
    f3_ax2.set_xticks([]), f3_ax2.set_yticks([])
    f3_ax3.set_xlim((0, 500))
    f3_ax3.set_xticks([])
    if i!=0:
        f3_ax3.set_yticks([])

fig.subplots_adjust(hspace=0, wspace=0)
"""
#%%
# try quick shock analysis
shocks_pos = []

for im in shocks:
    #im = shocks[14]
    
    nrows, ncols = im.shape
    x, y =np.arange(ncols), np.arange(nrows)
    X,Y = np.meshgrid(x,y)
    centroid = [np.sum(im*X)/np.sum(im), np.sum(im*Y)/np.sum(im)]
    
    channel_height = 30
    channel_width = 5
    
    data = np.mean( im[channel_height-channel_width:channel_height+channel_width, :], axis=0)
    data = np.gradient(data)
    nidx = np.argmin(data[20:35]) + 20
    
    #plt.figure()
    #plt.imshow(im)
    #plt.axvline(x=nidx)
    shocks_pos.append(nidx)

shocks_pos = np.array(shocks_pos)

#%%
# try quick focal spot analysis
    
focal_stats = []
for im in leakage_spots:
    #im = leakage_spots[16]
    im = np.array(im, dtype=float)
    
    im -= np.median(im) # bkg correct
    
    im[im<0] = 0
    
    
    nrows, ncols = im.shape
    x, y =np.arange(ncols), np.arange(nrows)
    X,Y = np.meshgrid(x,y)
    centroid = [np.sum(im*X)/np.sum(im), np.sum(im*Y)/np.sum(im)]
    
    xl = np.sum(im, axis=0)
    x0 = np.sum(x * xl) / np.sum(xl)
    yl = np.sum(im, axis=1)
    y0 = np.sum(y * yl) / np.sum(yl)
    
    ox = np.abs(np.nansum(xl * (x - x0)**2) / np.nansum(xl))**(0.5)
    oy = np.abs(np.nansum(yl * (y - y0)**2) / np.nansum(yl))**(0.5)
    
    X_ids = (X>=x0-ox) & (X<=x0+ox)
    Y_ids = (Y>=y0-oy) & (Y<=y0+oy)
    frac = np.sum(im * X_ids * Y_ids) / np.sum(im)
    # looks to be a good metric!
    
    o = (ox*oy)**(0.5)
    
    #plt.figure()
    #plt.imshow(im, origin='lower')
    #plt.plot(x0, y0, 'ro')
    #plt.axvline(x0 + ox), plt.axvline(x0 - ox)
    #plt.axhline(y0 + oy), plt.axhline(y0 - oy)
    focal_stats.append([x0, y0, ox, oy, o, frac])

focal_stats = np.array(focal_stats)

#%%
# quick espec analysis

def find_FWHM(x,y,frac=0.5):
    """Brute force FWHM calculation.
    Frac allows you to easily change, so to e-2 value etc.
    
    """
    fm = y.copy() - y.min()
    fm = fm.max()
    hm = fm * frac

    hm += y.min()
    fm = fm + y.min()
    max_idx = np.argmax(y)
    
    first_half = np.arange( int(0.9 * max_idx) )
    second_half = np.arange( int(1.1 * max_idx), x.size )
    
    hm_x_left = np.interp(hm, y[first_half], x[first_half])
    hm_x_right = np.interp(hm, y[second_half][::-1], x[second_half][::-1])
    
    return hm_x_right - hm_x_left

espec_stats = []

for spectrum in specs:
    #spectrum = specs[17]
    spectrum -= np.median(spectrum)
    spectrum[spectrum<0] = 0
    
    #E_min = 800
    #ids = E >= E_min
    #E0 = np.nansum(E[ids] * spectrum[ids]) / np.nansum(spectrum[ids])
    E0 = E[np.argmax(spectrum)]
    
    w = 500
    ids = (E<=E0 + w) & (E>=E0-w)
    
    oE = 2*w
    try:
        oE = np.abs(find_FWHM(E[ids],spectrum[ids]))
    except(ValueError):
        pass
    
    E_ids = (E>=E0-oE) & (E<=E0+oE)
    frac = np.nansum(spectrum * E_ids) / np.nansum(spectrum)
    espec_stats.append([E0, oE, frac])

    #plt.figure()
    #plt.plot(E, spectrum)
    #plt.axvline(x=E0, color='k')
    #plt.axvline(x=E0-oE, color='r'), plt.axvline(x=E0+oE, color='r')

    
    
espec_stats = np.array(espec_stats)

#%%


narrows = [15, 17, 22, 23, 32, 43, 55, 57,62, 72, 83, 90, 112, 113, 114, 115, 117, 122, 123, 143, 151]

narrows = np.array(narrows)

points = [shocks_pos, 
          focal_stats[:, -2], # o
          focal_stats[:, -1], # frac
          #espec_stats[:, 0], # E_max
          #1/espec_stats[:, -2]# width
          ]

fig, axes = plt.subplots(len(points),1, sharex=True)

for idx, l in enumerate(points):
    axes[idx].plot(l, 'x-')
    ps = l[narrows]
    axes[idx].plot(narrows, ps, 'kx')
    
    y0 = np.nanmean(ps)
    oy = np.nanstd(ps)#/len(ps)**(0.5)
    
    mini, maxi = np.nanmin(ps), np.nanmax(ps)
    #axes[idx].axhspan(ymin=mini, ymax=maxi, color='r', alpha=0.1)
    axes[idx].axhline(y=y0, color='k')
    axes[idx].axhspan(ymin=y0-oy, ymax=y0+oy, color='k', alpha=0.25)
    axes[idx].axhspan(ymin=y0-2*oy, ymax=y0+2*oy, color='k', alpha=0.1)
    

fig.subplots_adjust(hspace=0)




#%%
plt.figure()

for i in range(len(specs[100:])):
    spectrum = specs[i]
    spectrum -= np.nanmin(spectrum)
    spectrum /= np.nanmax(spectrum)
    
    spectrum /= 2.0
    
    plt.plot(spectrum + i, E, color='k')
    plt.axvline(x=i, color='grey')

plt.ylim((300, 1850))