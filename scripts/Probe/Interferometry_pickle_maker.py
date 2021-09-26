#Interferometry_Pickle_Maker.py
""" Script to run through to make calib ana_settings for LMI analysis
Copied entire workflow from 
RadiationReaction2021/ExampleNotebooks/Interferometry_Pickle_Maker
on experiment
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

import abel

#%%
# Choose a shot

diag = 'LMI'

# choose shot
#date = '20210621'
#run = 'run04' #run07

date = '20210609'
run = 'run11' #run07
shot_num = 32

shot_name = str(shot_num)
shot_name = 'Shot' + '0'*(3-len(shot_name)) + shot_name

# test path-finder function
I = Interferometry(None)
path = I.get_filepath([date, run, shot_num], diag=diag)
print(path)
#%%
# Get image

run_name = date + '/' + run
LMI = Interferometry(run_name, shot_num, cal_data_path=None)

plt.figure()
im = LMI.get_raw_image(path) # imitate what LivePLotting will give it here
plt.xlabel('pixels'), plt.ylabel('pixels')
xmin, xmax, ymax, ymin = 0, 656, 492, 0
plt.imshow(im, extent=[xmin, xmax, ymax, ymin])
plt.title('raw image - %s %s %s' % (date, run, shot_name))

#%%
# Set fringes region of interest
#[top, bottom, left, right] = [235, 280, 180, 460]
[top, bottom, left, right] = [245, 282, 245, 465]

LMI.fringes_roi = [top, bottom, left, right]
LMI.fringes_img_extent = [left, right, bottom, top]

plt.figure()
im = LMI.get_raw_image(path) # imitate what LivePLotting will give it here
plt.xlabel('pixels'), plt.ylabel('pixels')
plt.imshow(im, extent=LMI.raw_img_extent)
plt.title('ROI - %s %s %s' % (date, run, shot_name))

t,b,l,r = LMI.fringes_roi
plt.plot([l, r, r, l, l], [t, t, b, b, t], 'y-')


plt.figure()
Z = LMI.get_fringe_image(LMI.raw_img)
plt.imshow(Z, extent=LMI.fringes_img_extent)
plt.title('fringe_image (ROI only)')


#%%
# Set references and mask in fourier space
ref_raw_imgs = None

#ref_date = '20210621'
#ref_run = 'run06'
#ref_shots = np.arange(1,10+1)

ref_date = '20210609'
ref_run = 'run05'
ref_shots = np.arange(41, 53)


l = list(ref_shots)
l = [[ref_date, ref_run, bs] for bs in l]

LMI.ref_shots = l
ref_raw_imgs = LMI.get_ref_images(l)


Z = LMI.get_fringe_image(LMI.raw_img)
U, V, F_Z = LMI.FFT(Z)

fig, axes = plt.subplots(3,2)
axes[0,0].imshow(Z, extent=LMI.fringes_img_extent)

xmin, xmax, ymax, ymin = U.min(), U.max(), V.min(), V.max()
axes[0,1].imshow(np.log10(np.abs(F_Z)),  extent=[xmin, xmax, ymax, ymin])

#uu, ou = 75.0, 10.0 #51.0, 3
#uv, ov = 62., 20.0
uu, ou = -29.5, 3.5 #51.0, 3
uv, ov = 0., 20.0
gamma_u, gamma_v = 5., 5.

fourier_mask = LMI.Gauss(U, V, uu, uv, ou, ov, gamma_u, gamma_v)
F_Z_masked = F_Z * fourier_mask

# try something new
"""
uu,uv = 0.0, 0.0
R = ((U-uu)**2 + (V - uv)**2)**(0.5)
oR = 10.0
uR = 40.0
gamma_R = 5.

# half-donut
R[U>0.0] =  0.0

#R = ((U-uR)**2 + (V - uR)**2)**(0.5)
fm =  np.exp(-((R-uR)**2/oR**2)**gamma_R)
u = U[0]
v = V[:,0]
fm /= np.sum(fm * np.gradient(u)[np.newaxis,:] * np.gradient(v)[:,np.newaxis])
fourier_mask = np.copy(fm)
F_Z_masked = F_Z * fourier_mask
"""

F_Z_masked_to_plot = F_Z.copy()
F_Z_masked_to_plot[fourier_mask == 0.0] = np.nan
xmin, xmax, ymax, ymin = U.min(), U.max(), V.min(), V.max()
axes[1,1].imshow(np.log10(np.abs(F_Z_masked_to_plot)), extent=[xmin, xmax, ymax, ymin])
LMI.fourier_mask = fourier_mask


# check refs
if ref_raw_imgs is not None:
    ref_fringes_img, U, V, ref_F_Z, ref_F_Z_masked  = LMI.check_ref_images(LMI.fringes_img, ref_raw_imgs = ref_raw_imgs)
    fig_ref, axes_ref = plt.subplots(3,1)
    axes_ref[0].set_title("Average of background references")
    axes_ref[0].imshow(ref_fringes_img, extent=LMI.fringes_img_extent)
    xmin, xmax, ymax, ymin = U.min(), U.max(), V.min(), V.max()
    axes_ref[1].imshow(np.log10(np.abs(ref_F_Z)),  extent=[xmin, xmax, ymax, ymin])
    ref_F_Z_masked_to_plot = ref_F_Z.copy()
    ref_F_Z_masked_to_plot[fourier_mask == 0.0] = np.nan
    axes_ref[2].imshow(np.log10(np.abs(ref_F_Z_masked_to_plot)),  extent=[xmin, xmax, ymax, ymin])

axes[2,1].imshow(fourier_mask, extent=[xmin, xmax, ymax, ymin])
phase = LMI.calc_phase(LMI.fringes_img, shift_in=True, shift_out=True, ref_raw_imgs = ref_raw_imgs)
print(LMI.fringes_img_extent)
axes[1,0].imshow(phase, extent=LMI.fringes_img_extent)
xmin, xmax, ymax, ymin = LMI.fringes_img_extent
axes[2,0].plot(np.arange(xmin, xmax), np.mean(phase, axis=0))
axes[2,0].set_xlim((xmin, xmax))

axes[0,0].set_title('Real Space')
axes[0,1].set_title('Fourier Space')


#%%
plt.figure()
sig = np.copy(LMI.fringes_img)
bkg = np.copy(ref_fringes_img)
#sig -= sig.min()
#sig /= sig.max()
#bkg -= np.min(bkg)
#bkg /= bkg.max()

plt.title('Spatial difference - Channel outline')

diff = np.abs(sig - bkg)

plt.imshow(diff, extent=LMI.fringes_img_extent)

#%%
# Check channel ROI

# top, bottom, left, right format
channel_roi = [253,281, 274, 438]
LMI.channel_roi = channel_roi

Z = np.copy(LMI.phase)

plt.figure()
plt.title('Channel ROI')
cax = plt.imshow(Z, extent=LMI.fringes_img_extent)
t,b,l,r = LMI.channel_roi
plt.plot([l, r, r, l, l], [t, t, b, b, t], 'y-')
cbar = plt.colorbar(cax)
cbar.set_label('Phase [rad]')

#%%
# Additional correction (if no refs provided)

LMI.apply_no_ref_correction = False

channel_roi = np.copy(LMI.channel_roi)
LMI.calc_channel_mask()

fig, axes = plt.subplots(2,2)
Z = np.copy(LMI.fringes_img)
ROI = channel_roi
#Z[ROI[0]:ROI[1], ROI[2]:ROI[3]] = np.nan
axes[0,0].imshow(Z)

phase_cp = np.copy(LMI.phase)

if LMI.apply_no_ref_correction:
    (x_axis, x_trend, x_correction, y_axis, y_trend, y_correction), phase_bg_corr = LMI.no_ref_correction(overwrite=True)
else:
    (x_axis, x_trend, x_correction, y_axis, y_trend, y_correction), phase_bg_corr = LMI.no_ref_correction(overwrite=False)
    
axes[0,1].set_title('x trend')
axes[0,1].plot(x_axis, x_trend)
axes[0,1].plot(x_axis, x_correction, 'r--')

axes[1,1].set_title('then y trend')
axes[1,1].plot(y_axis, y_trend)
axes[1,1].plot(y_axis, y_correction, 'r--')

axes[1,0].set_title('Fringe shift (2$\pi$)')
cax = axes[1,0].imshow(phase_bg_corr/(2.0*np.pi))

#%%
# Polarity flipper
#override here if necessary

LMI.invert_phase = True

# plot initial
fig, axes = plt.subplots(2,1)

p = np.copy(LMI.phase)
axes[0].imshow(p, extent=LMI.fringes_img_extent)
t,b,l,r = LMI.channel_roi
axes[0].plot([l, r, r, l, l], [t, t, b, b, t], 'y-')
axes[0].set_title('Original Phase')

if LMI.invert_phase == True:
    LMI.phase = -1.0 * LMI.phase

else:
    # test out polarity finder
    LMI.channel_roi = channel_roi

    channel_roi = np.copy(LMI.channel_roi)
    LMI.calc_channel_mask()


    phase = LMI.phase
    c_mask = LMI.channel_mask
    not_c_mask = np.where(c_mask==1.0, 0.0, 1.0)

    inner = np.nanmean(phase * c_mask)
    outer = np.nanmean(phase * not_c_mask)

    inner = np.nanmean(phase * c_mask, axis=1)
    inner = inner[inner != 0.]
    inner = np.nanmean(inner)

    outer = np.nanmean(phase * not_c_mask, axis=1)
    outer = outer[outer != 0.]
    outer = np.nanmean(outer)

    print(inner, outer)
    #print(inner < outer)

    """
    plt.figure()
    inner = np.nanmean(phase * c_mask, axis=0)
    outer = np.nanmean(phase * not_c_mask, axis=0)
    plt.plot(inner), plt.plot(outer)
    inner = np.nanmean(phase * c_mask, axis=1)
    outer = np.nanmean(phase * not_c_mask, axis=1)
    plt.plot(inner), plt.plot(outer)
    """

    LMI.correct_phase_polarity()

#plot final
p = np.copy(LMI.phase)
axes[1].imshow(p, extent=LMI.fringes_img_extent)
axes[1].set_title('Inverted Phase')

#%%
# Angle Fitter

fig, axes = plt.subplots(2,1)
phase_bg_corr = np.copy(LMI.phase)
c_mask = np.copy(LMI.channel_mask)
p = np.copy(phase_bg_corr) * c_mask
p -= np.nanmean(p)
axes[0].imshow(p)
axes[0].set_title('Channel Angle Guess')
"""
t,b,l,r = LMI.fringes_roi
x0 = np.arange(l,r) - l
y0 = np.nanmean(p, axis=0)
axes[0].plot(x0, y0, 'x')
"""

n_std_threshold = 0.7
LMI.n_std_threshold = np.copy(n_std_threshold)
theta, c, x, yc = LMI.calc_channel_angle()
axes[0].plot(x, yc, 'x')
m = np.tan(theta * np.pi/180.0)
coeff = [m, c]
poly = np.poly1d(coeff)
axes[0].plot(x, poly(x), 'r')

print('theta: ', theta)

phase = LMI.rotate_image(LMI.phase, theta)
LMI.phase = phase
axes[1].imshow(phase)
axes[1].set_title('Angle Corrected Channel')

#%%
# Abel Invert

# ABEL INVERT
LMI.umperpixel = 83.0

LMI.centre_method = ['convolution', 'gaussian']
phase_original = np.copy(LMI.phase)
#centre = LMI.calc_phase_centre()

centre = (110.5, 24.975388419472814)
LMI.fixed_channel_centre = centre
LMI.centre = centre

# check centering
phase = phase_original.T
shift = 0
#centre = (centre[0], centre[1] + shift)



if abel._version == '0.8.4': 
    centred_image = abel.tools.center.set_center(phase, origin=centre, axes=1)
else:
    #centre = (centre[1], centre[0]) # annoying correction in using OLD abel - this correction seems wrong?
    centred_image = abel.tools.center.set_center(phase, center=centre)#, axes=1)
centred_image = centred_image.T



#LMI.phase = np.copy(centred_image) # use if changed inversion axis from what was found
# get number density
LMI.phase_to_SI()
ne = LMI.abel_invert_int_ne_dl(method='direct')
#ne = LMI.get_ne(path).T
nrows, ncols = ne.shape
mid = nrows//2

fig, axes = plt.subplots(3,1)
axes[0].imshow(phase_original, extent=LMI.fringes_img_extent), axes[0].set_title('Raw Phase')
axes[1].imshow(centred_image[:mid,:]), axes[1].set_title('Top')
axes[2].imshow(centred_image[mid:, :]), axes[2].set_title('Bottom')


plt.figure()
cax = plt.imshow(ne, extent=LMI.fringes_img_extent)
plt.title('Abel-Inverted')
cbar = plt.colorbar(cax)
cbar.set_label('$n_e$ [cm$^{-3}$]')

plt.figure()
channel_width = 8
LMI.channel_width = channel_width
hw = channel_width // 2
x = np.arange(ncols) * LMI.umperpixel * 1e-3
plt.plot(x, np.mean( ne[mid-hw:mid, :], axis=0 ), '--', label='top')
plt.plot(x, np.mean( ne[mid:mid+hw, :], axis=0 ), '--', label='bottom')
plt.plot(x, np.mean( ne[mid-hw:mid+hw, :], axis=0 ), label='both')
plt.xlabel('$x$ [mm]')
plt.title('Average over channel width')
plt.legend()
plt.grid()
plt.ylabel('$n_e$ [cm$^{-3}$]')


#%%
plt.show()


#%%
# Save settings
run_valid_from = 'run11'
shot_valid_from = 1


run_name = date + '/' + run_valid_from
LMI.run_name = run_name
LMI.shot_num = shot_valid_from
LMI.cal_data_path = HOME + '/calib/Probe'
filename_to_save = HOME + '/calib/Probe' + '/' + LMI.diag + '/'
date, run = LMI.run_name.split('/')
shot_num = str(shot_valid_from)
ss = 'shot' + '0'*(3 - len(shot_num)) + shot_num #LOWERCASE s
filename_to_save += '_ana_settings_' + date + '_' + run + '_' + ss + '.pkl'

ans = input('Save pickle file of settings.\nFile will be %s - y or n? ' % (filename_to_save))

if ans=='y':
    LMI.save_ana_settings()


