#F2_focal_spot_analysis.py

import sys
sys.path.append('../../')
from setup import *
from lib.pipeline import *
from lib.general_tools import *
from modules.FocalSpot.FocalSpot import *
from modules.FocalSpot.F2_date_to_properties_dicts import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c

def get_img(img):
    return img

#'20210603', '20210604', '20210614', '20210617', '20210618', '20210620', '20210621', '20210622'

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

#read in data
f2_pipeline= DataPipeline(diag_f2, north_beam.focal_spot.get_spot_properties_lst_sqrd_fit, single_shot_mode=False)
shot_num_f2, f2_data = f2_pipeline.run('%s/%s'%(date, run))
shot_num_f2=np.array(shot_num_f2)

#calculate mean & std focal spot properties for each burst
burst_nos=np.unique(shot_num_f2[:, 0])
mean_params=np.array([np.mean(f2_data[np.where(shot_num_f2[:, 0]==burst)], axis=0) for burst in burst_nos])
std_params=np.array([np.std(f2_data[np.where(shot_num_f2[:, 0]==burst)], axis=0) for burst in burst_nos])

# get positions of F2 focal cam from dict
f2_cam_positions=np.array(list(map(dict[int(date)].get, burst_nos)))

#fit laser waist to focal spot data
popt_ax1, pcov_ax1= opt.curve_fit(north_beam.calc_waist, f2_cam_positions, mean_params[:, 3]*2.0, p0=[1.0, 1.0, f2_cam_positions[0]], maxfev=5000, sigma=std_params[:, 3]*2.0)#, args=(f2_cam_positions[0], l0, n))
popt_ax2, pcov_ax2= opt.curve_fit(north_beam.calc_waist, f2_cam_positions, mean_params[:, 4]*2.0, p0=[1.0, 1.0, f2_cam_positions[0]], maxfev=5000, sigma=std_params[:, 4]*2.0)#, args=(f2_cam_positions[0], l0, n))

print(convert_width_to_FHWM(popt_ax1[0]))
print(convert_width_to_FHWM(popt_ax2[0]))
print(popt_ax1[1])
print(popt_ax2[1])
print(popt_ax1[2])
print(popt_ax2[2])

# find measured focal position closest to least squares fitted focal position
indexes_focal_plane=np.argmin(mean_params[:, 3])

#update north beam with focal spot properties
north_beam.focal_spot=FocalSpot(focal_pos_z=popt_ax1[2], focal_pos_z_err=np.sqrt(np.diag(pcov_ax1))[2], energy_frac_FWHM=mean_params[indexes_focal_plane, 7], energy_frac_FWHM_err=std_params[indexes_focal_plane, 7], FWHM_x=convert_width_to_FHWM(mean_params[indexes_focal_plane, 3]*2.0), FWHM_x_err=convert_width_to_FHWM(std_params[indexes_focal_plane, 3]*2.0), FWHM_y=convert_width_to_FHWM(mean_params[indexes_focal_plane, 4]*2.0), FWHM_y_err=convert_width_to_FHWM(std_params[indexes_focal_plane, 4]*2.0), microns_per_pixel=microns_per_pixel, microns_per_pixel_err=microns_per_pixel_err)

waist_fitted_ax1=north_beam.calc_waist(f2_cam_positions, popt_ax1[0], popt_ax1[1], popt_ax1[2])
waist_fitted_ax2=north_beam.calc_waist(f2_cam_positions, popt_ax2[0], popt_ax2[1], popt_ax2[2])

plt.scatter(f2_cam_positions, mean_params[:, 3]*2.0, color='b')
plt.scatter(f2_cam_positions, mean_params[:, 4]*2.0, color='cyan')
plt.scatter(f2_cam_positions, waist_fitted_ax1, color='r')
plt.scatter(f2_cam_positions, waist_fitted_ax2, color='g')
plt.vlines(f2_cam_positions, mean_params[:, 3]*2.0-std_params[:, 3]*2.0, mean_params[:, 3]*2.0+std_params[:, 3]*2.0, color='b', alpha=0.4)
plt.vlines(f2_cam_positions, mean_params[:, 4]*2.0-std_params[:, 4]*2.0, mean_params[:, 4]*2.0+std_params[:, 4]*2.0, color='cyan', alpha=0.4)
plt.show()

peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_err=north_beam.calc_peak_intensity()
north_beam.a0, north_beam.a0_err=north_beam.calc_a0()

print('energy_frac_fhwm=%s+/-%s'%(north_beam.focal_spot.energy_frac_FWHM, north_beam.focal_spot.energy_frac_FWHM_err))
print('FWHM_ax1_fitted microns=%s+/-%s'%(north_beam.focal_spot.FWHM_x, north_beam.focal_spot.FWHM_x_err))
print('FWHM_ax2_fitted microns=%s+/-%s'%(north_beam.focal_spot.FWHM_y, north_beam.focal_spot.FWHM_y_err))
print('peak_intensity_W_per_cm2=%s+/-%s'%(peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_err))
print('a0=%s+/-%s'%(north_beam.a0, north_beam.a0_err))
