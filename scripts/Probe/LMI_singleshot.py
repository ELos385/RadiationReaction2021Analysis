import sys
sys.path.append('../../')
from setup import *
from modules.Probe.interferometry import *
import numpy as np
import matplotlib.pyplot as plot

''' User inputs '''

# shot
date = '20210620'
run = 'Set1_nulls_20210620'
shot_num = 0
diag = 'LMI'

# reference
date_ref = '20210621'
run_ref = 'run06'
shot_num_ref = [1,2,3,5,6,7,8,9,10]

# regions of interest
fringes_roi = [220, 280, 155, 465] # vertical, horizontal
channel_roi = [35, 55, 30, 290]

retrieval_method = 'fft'  # hilbert or fft

um_per_px = 84.3

# mask for fft retrieval
ox = 16
mask_params = [113, 30, ox, 20, 8, 8]   #ux, uy, wx, wy, nx, ny

''' Run the code '''

I = interferometry(None)
path = I.get_filepath([date, run, shot_num])

run_name = date + '/' + run
run_name_ref = date_ref + '/' + run_ref

# Interferometry object

LMI = interferometry(run_name, shot_num, cal_data_path=None)

LMI.fringes_roi = fringes_roi
LMI.channel_roi = channel_roi
LMI.umperpixel = um_per_px
LMI.fmask_params = mask_params

#ref shots
LMI.ref_shots = [[date_ref, run_ref, bs] for bs in shot_num_ref]

# density
ne, n_noise = LMI.get_ne(path=path, retrieval_method=retrieval_method,
    calc_centre_method=['convolution', 'gaussian'], calc_error = True)

# display result
nrows, ncols = ne.shape
mid = nrows // 2
x = np.arange(ncols) * um_per_px * 1e-3
y = np.linspace(-mid, mid, nrows) * um_per_px * 1e-3

plt.figure()
plt.imshow(ne, extent=[x[0], x[-1], y[0], y[-1]], vmin=0)
plt.xlabel('x (mm)')
plt.ylabel('r (mm)')
#plt.colorbar()
#plt.tight_layout()

# Density lineout
channel_width = 8
hw = channel_width // 2
x = np.arange(ncols) * LMI.umperpixel * 1e-3
y = np.arange(nrows) * LMI.umperpixel * 1e-3

# Calculate the uncertainties
mean_density = np.mean( ne[mid-hw:mid+hw, :], axis=0)
top = np.mean( ne[mid-hw:mid, :], axis=0 )
bottom = np.mean( ne[mid:mid+hw, :], axis=0)
lineout_error = np.abs(top - bottom)/2
total_error = np.sqrt(lineout_error**2 + n_noise**2)

# Plot the density lineout
plt.figure()
plt.plot(x, mean_density, label='average')
plt.fill_between(x, mean_density + total_error, mean_density - total_error,
    alpha=0.5, label='1$\sigma$')
plt.xlabel('x (mm)')
plt.ylabel('n$_e$ (cm$^{-3}$)')
plt.ylim([-0.5e18, 2e18])
plt.title('%s %s shot %s'%(date,run,shot_num))
plt.legend()
plt.show()
