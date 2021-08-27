#F2_plot_focal_spot_and_fitted_spot.py

import sys
sys.path.append('../../')
from setup import *
from lib.pipeline import *
from lib.general_tools import *
from modules.FocalSpot.FocalSpot import *
from modules.FocalSpot.F2_date_to_properties_dicts import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_img(img):
    return img

date= '20210618'
run=F2_date_to_run_dict[int(date)]
diag_f2='f2focus'
#load from dict (or pickle?), according to date and diag name
wavelength=0.8#microns
refractive_index=1.0
f_number=2.0

#get laser properties from dicts according to date
microns_per_pixel, microns_per_pixel_err=F2_date_to_spatial_calibration_dict[int(date)][0], F2_date_to_spatial_calibration_dict[int(date)][1]
FWHM_t, FWHM_t_err=F2_date_to_FWHM_t_dict[int(date)][0], F2_date_to_FWHM_t_dict[int(date)][1]
energy, energy_err=F2_date_to_north_energy_dict[int(date)][0], F2_date_to_north_energy_dict[int(date)][1]
throughput, throughput_err=F2_date_to_north_thoughput_dict[int(date)][0], F2_date_to_north_thoughput_dict[int(date)][1]

#initialise laser object
north_beam=Laser(wavelength, refractive_index, FWHM_t, FWHM_t_err, f_number, energy, energy_err, throughput, throughput_err, microns_per_pixel=microns_per_pixel, microns_per_pixel_err=microns_per_pixel_err)

f2img_pipeline= DataPipeline(diag_f2, get_img, single_shot_mode=False)
shot_num_f2, f2_img = f2img_pipeline.run('%s/%s'%(date, run))

spot_props=np.array(north_beam.focal_spot.get_spot_properties_lst_sqrd_fit(f2_img[0])).reshape(-1, 8)
spot_props[:,1:5]=spot_props[:,1:5]/microns_per_pixel

x_max=len(f2_img[0][0])#*pix_size_x/mag
y_max=len(f2_img[0])
x = np.linspace(0, x_max, x_max)#*microns_per_pixel
y = np.linspace(0, y_max, y_max)#*microns_per_pixel
X, Y = np.meshgrid(x, y)

print('%s, %s, %s, %s, %s, %s, %s'%(spot_props[i,0], spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5], spot_props[i,6]))
fitted_gauss=two_d_gaussian([X, Y], spot_props[i,0], spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5], spot_props[i,6]).reshape(X.shape)
g=calc_ellipse(X, Y, spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5])
y_ell=Y[np.where(g<1.0)]
x_ell=X[np.where(g<1.0)]

fig, ax = plt.subplots(2)
p1=ax[0].imshow(f2_img[i], vmin=np.amin(f2_img[i]), vmax=np.amax(f2_img[i]), origin='lower')
ax[0].set_aspect('auto')
ax[0].scatter(x_ell, y_ell, color='r')
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(p1, cax=cax1)
#plt.scatter(centre_x_pos[i], centre_y_pos[i])
p2=ax[1].pcolor(X, Y, fitted_gauss, vmin=np.amin(f2_img[i]), vmax=np.amax(f2_img[i]))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(p2, cax=cax2)
plt.show()
