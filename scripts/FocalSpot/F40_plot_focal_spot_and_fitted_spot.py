#F2_plot_focal_spot_and_fitted_spot.py

import sys
sys.path.append('../../')
from setup import *
from lib.pipeline import *
from lib.general_tools import *
from modules.FocalSpot.FocalSpot import *
from modules.FocalSpot.F40_date_to_properties_dicts import *
# from .loader import register_data_loader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('/Users/ee.los/.matplotlib/mpl_configdir/thesis.mplstyle')


def get_img(img):
    return img

date= '20210608'
run=F40_date_to_run_dict[int(date)]
diag_f40='F40focus'
#load from dict (or pickle?), according to date and diag name
wavelength=0.8#microns
refractive_index=1.0
f_number=40.0

#get laser properties from dicts according to date
microns_per_pixel, microns_per_pixel_err=F40_date_to_spatial_calibration_dict[int(date)][0], F40_date_to_spatial_calibration_dict[int(date)][1]
FWHM_t, FWHM_t_err=F40_date_to_FWHM_t_dict[int(date)][0], F40_date_to_FWHM_t_dict[int(date)][1]
energy, energy_err=F40_date_to_south_energy_dict[int(date)][0], F40_date_to_south_energy_dict[int(date)][1]
throughput, throughput_err=F40_date_to_south_thoughput_dict[int(date)][0], F40_date_to_south_thoughput_dict[int(date)][1]

#initialise laser object
south_beam=Laser(wavelength, refractive_index, FWHM_t, FWHM_t_err, f_number, energy, energy_err, throughput, throughput_err, microns_per_pixel=microns_per_pixel, microns_per_pixel_err=microns_per_pixel_err)

f40img_pipeline= DataPipeline(diag_f40, get_img, single_shot_mode=False)
shot_num_f40, f40_img = f40img_pipeline.run('%s/%s'%(date, run))
print(shot_num_f40)
burst_nos=np.array(list(date_to_burst_and_position[int(date)].keys()))
f40_cam_positions=np.array(list(map(date_to_burst_and_position[int(date)].get, burst_nos)))
f40_cam_positions, burst_nos = (list(t) for t in zip(*sorted(zip(f40_cam_positions, burst_nos))))
f40_cam_positions, burst_nos=np.array(f40_cam_positions), np.array(burst_nos)
print(burst_nos)

# no_shots=len(shot_num_f40)/len(burst_nos)

burst_no_lower=0#len(burst_nos)
burst_no_higher=5#len(burst_nos)
f40_cam_positions=f40_cam_positions[burst_no_lower:burst_no_higher]
burst_nos=burst_nos[burst_no_lower:burst_no_higher]
print(burst_nos)
burst_no_range=burst_no_higher-burst_no_lower
av_spot_props=np.zeros((burst_no_range, 8))
std_spot_props=np.zeros((burst_no_range, 8))
example_spot_props=np.zeros((burst_no_range, 8))
plt_img_arr=np.zeros((burst_no_range, f40_img[0].shape[0], f40_img[0].shape[1]))
c=0
tmp_lst=[]
for i in range(len(shot_num_f40)):
    if shot_num_f40[i][0] in burst_nos:
        c=np.where(shot_num_f40[i][0]==burst_nos)
        print("c=%s"%c)
        print("burst_nos[c]=%s"%burst_nos[c])
        print("shot no indxed = %s"%shot_num_f40[i][0])
        # print("burst no arr = %s"%burst_nos[c])
        if (shot_num_f40[i][0]==burst_nos[c]):
            print("burst_nos[c] in statement 1=%s"%burst_nos[c])
            # print("burst_nos[c] 1 = %s"%burst_nos[c])
            spot_props=np.array(south_beam.focal_spot.get_spot_properties_lst_sqrd_fit(f40_img[i])).reshape(8)
            tmp_lst.append(spot_props)
            # print("c 1=%s"%c)
            # print("print(tmp_lst1)=%s"%tmp_lst)
            # print("print(tmp_arr1)=%s"%np.array(tmp_lst))
            if (shot_num_f40[i][1]==1):
                plt_img_arr[c, :, :]=f40_img[i]
                example_spot_props[c, :]=spot_props

            if (len(shot_num_f40)-1==i or shot_num_f40[i+1][0]>shot_num_f40[i][0]):
                # print('here')
                tmp_arr=np.asarray(tmp_lst)
                # print("print(tmp_arr)=%s"%tmp_arr)
                # print(tmp_arr.shape)
                av_spot_props[c, :]=np.mean(tmp_arr, axis=0)
                std_spot_props[c, :]=np.std(tmp_arr, axis=0)
                tmp_lst=[]
                spot_props=np.array(south_beam.focal_spot.get_spot_properties_lst_sqrd_fit(f40_img[i])).reshape(8)
                tmp_lst.append(spot_props)
            # print("print(tmp_lst)=%s"%tmp_lst)
        else:
            tmp_arr=np.asarray(tmp_lst)
            # print("print(tmp_arr)=%s"%tmp_arr)
            # print("burst_nos[c] 2 = %s"%burst_nos[c])
            # print(tmp_arr.shape)
            # print("c=%s"%c)
            av_spot_props[c, :]=np.mean(tmp_arr, axis=0)
            std_spot_props[c, :]=np.std(tmp_arr, axis=0)
            tmp_lst=[]
            spot_props=np.array(south_beam.focal_spot.get_spot_properties_lst_sqrd_fit(f40_img[i])).reshape(8)
            tmp_lst.append(spot_props)
            c+=1
            if (shot_num_f40[i][1]==1):
                plt_img_arr[c, :, :]=f40_img[i]
                example_spot_props[c, :]=spot_props
    else:
        # print(burst_nos[c])
        continue

print('av_spot_props=%s'%av_spot_props)
print('std_spot_props=%s'%std_spot_props)
print(plt_img_arr)
fig, axs = plt.subplots(2, len(burst_nos), figsize=(12, 4), constrained_layout=True)
ax=axs.flatten()
xmin=200
xmax=len(f40_img[0][0])-60
ymin=80
ymax=390
for i in range(len(burst_nos)):
    plt_img_temp=plt_img_arr[i, ymin:ymax, xmin:xmax]
    x_max=len(plt_img_temp[0])#*pix_size_x/mag
    y_max=len(plt_img_temp)
    # x = np.linspace(xmin, xmax, xmax-xmin)#*microns_per_pixel
    # y = np.linspace(ymin, ymax, ymax-ymin)#*microns_per_pixel
    print(x_max)
    print(y_max)
    x = np.linspace(0, x_max, x_max)#*microns_per_pixel
    y = np.linspace(0, y_max, y_max)#*microns_per_pixel
    X, Y = np.meshgrid(x, y)
    print(X.shape)
    print(Y.shape)
    example_spot_props=np.array(south_beam.focal_spot.get_spot_properties_lst_sqrd_fit(plt_img_temp)).reshape(8)
    fitted_gauss=two_d_gaussian([X, Y], example_spot_props[0], example_spot_props[1], example_spot_props[2], example_spot_props[3], example_spot_props[4], example_spot_props[5], example_spot_props[6])
    print(fitted_gauss)
    print(fitted_gauss.shape)
    fitted_gauss=fitted_gauss.reshape(X.shape)
    g=calc_ellipse(X, Y, example_spot_props[1], example_spot_props[2], example_spot_props[3], example_spot_props[4], example_spot_props[5])

    # fitted_gauss=two_d_gaussian([X, Y], example_spot_props[i,0], example_spot_props[i,1], example_spot_props[i,2], example_spot_props[i,3], example_spot_props[i,4], example_spot_props[i,5], example_spot_props[i,6]).reshape(X.shape)
    # g=calc_ellipse(X, Y, example_spot_props[i,1], example_spot_props[i,2], example_spot_props[i,3], example_spot_props[i,4], example_spot_props[i,5])
    y_ell=Y[(g<1.2) & (g>0.98)]
    x_ell=X[(g<1.2) & (g>0.98)]
    ax[i].imshow(plt_img_temp, vmin=np.min(plt_img_arr), vmax=np.max(plt_img_arr))
    ax[i+len(burst_nos[0:burst_no_range])].imshow(fitted_gauss, vmin=np.min(plt_img_arr), vmax=np.max(plt_img_arr))
    im1= ax[i].scatter(x_ell, y_ell, color='r', s=0.5)
    # ax[i+len(burst_nos[0:burst_no_range])].set_xlabel("AO focal term")
    im2=ax[i+len(burst_nos[0:burst_no_range])].axes.xaxis.set_ticks([int(x_max/2.0)])
    ax[i+len(burst_nos[0:burst_no_range])].axes.xaxis.set_ticklabels(["AO focal term=%s"%f40_cam_positions[i]])
    ax[i].axes.xaxis.set_ticks([])
    ax[i].axes.xaxis.set_ticklabels([])
    if i==0:
        ax[i+len(burst_nos)].axes.yaxis.set_ticks([0])
        ax[i+len(burst_nos)].axes.yaxis.set_ticklabels(["Fitted focal spot"], rotation = 90)
        ax[i].axes.yaxis.set_ticks([0])
        ax[i].axes.yaxis.set_ticklabels(["Measured focal spot"], rotation = 90)
    else:
        ax[i+len(burst_nos)].axes.yaxis.set_ticks([])
        ax[i+len(burst_nos)].axes.yaxis.set_ticklabels([])
        ax[i].axes.yaxis.set_ticks([])
        ax[i].axes.yaxis.set_ticklabels([])
        # ax[i+len(burst_nos[0:burst_no_range])].tick_params(axis='y', colors='white')
        # ax[i].tick_params(axis='y', colors='white'2

# plt.subplots_adjust(right=0.8)
# fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.subplots_adjust(bottom=0.1, top=1.0, left=0.05, right=0.9,
                    wspace=0.0, hspace=0.0)

# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.85])
cbar = fig.colorbar(im1, cax=cb_ax)

# fig.colorbar(im1, ax=axs.ravel().tolist())
# divider = make_axes_locatable(ax[len(burst_nos)-1])
# cax1 = divider.append_axes('right', size='8%', pad=0.05)
# fig.colorbar(im1, cax=cax1)
# print(len(ax))
# print(len(burst_nos)-1)
# print(burst_no_range*2.0-1)
# divider = make_axes_locatable(ax[int(burst_no_range*2.0-1)])
# cax2 = divider.append_axes('right', size='8%', pad=0.05)
# fig.colorbar(im1, cax=cax2)
# cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
# fig.colorbar(im, cax=cbar_ax)
# plt.tight_layout()
plt.show()

#fig, axs = plt.subplots(1, figsize=(10, 4))
# for i in range(burst_no_range):

indexes_focal_plane=np.argmin(av_spot_props[:, 3])
print("indexes_focal_plane = %s"%indexes_focal_plane)
# f40_cam_positions=np.array(list(map(date_to_burst_and_position[int(date)].get, burst_nos)))[burst_no_lower:burst_no_higher]

#update south beam with focal spot properties
popt_ax1, pcov_ax1= opt.curve_fit(south_beam.calc_waist, f40_cam_positions, av_spot_props[:, 3]*2.0, p0=[1.0, 1.0, f40_cam_positions[0]], bounds=(0.0, 100.0), maxfev=5000, sigma=std_spot_props[:, 3]*2.0)#, args=(f2_cam_positions[0], l0, n))
popt_ax2, pcov_ax2= opt.curve_fit(south_beam.calc_waist, f40_cam_positions, av_spot_props[:, 4]*2.0, p0=[1.0, 1.0, f40_cam_positions[0]], bounds=(0.0, 100.0), maxfev=5000, sigma=std_spot_props[:, 4]*2.0)#, args=(f2_cam_positions[0], l0, n))
waist_x=south_beam.calc_waist(f40_cam_positions, popt_ax1[0], popt_ax1[2], popt_ax1[2])
waist_y=south_beam.calc_waist(f40_cam_positions, popt_ax2[0], popt_ax2[2], popt_ax2[2])
frac_err_popt2=np.sqrt((np.sqrt(np.diag(pcov_ax2))[0]/popt_ax2[0])**2+(np.sqrt(np.diag(pcov_ax2))[1]/popt_ax2[1])**2+(np.sqrt(np.diag(pcov_ax2))[2]/popt_ax2[2])**2)
frac_err_popt1=np.sqrt((np.sqrt(np.diag(pcov_ax1))[0]/popt_ax1[0])**2+(np.sqrt(np.diag(pcov_ax1))[1]/popt_ax1[1])**2+(np.sqrt(np.diag(pcov_ax1))[2]/popt_ax1[2])**2)
# print('np.sqrt(np.diag(pcov_ax1))[2]=%s'%np.sqrt(np.diag(pcov_ax1))[2])
# print("frac_err_popt2 = %s"%frac_err_popt2)
# print("frac_err_popt1 = %s"%frac_err_popt1)
# print("pcov_ax2[0] = %s"%pcov_ax2[0])
# print("np.sqrt(np.diag(pcov_ax2))[0] = %s"%np.sqrt(np.diag(pcov_ax2))[0])

print("(np.sqrt(np.diag(pcov_ax2))[0]/pcov_ax2[0])**2=%s"%(np.sqrt(np.diag(pcov_ax2))[0]/pcov_ax2[0])**2)

print("popt_ax1 = %s"%popt_ax1)
print("np.sqrt(np.diag(pcov_ax1))=%s"%np.sqrt(np.diag(pcov_ax1)))
print("popt_ax2 = %s"%popt_ax2)
print("np.sqrt(np.diag(pcov_ax2))=%s"%np.sqrt(np.diag(pcov_ax2)))

south_beam.focal_spot=FocalSpot(focal_pos_z=popt_ax1[2], focal_pos_z_err=np.sqrt(np.diag(pcov_ax1))[2], energy_frac_FWHM=av_spot_props[indexes_focal_plane, 7], energy_frac_FWHM_err=std_spot_props[indexes_focal_plane, 7], FWHM_x=convert_width_to_FHWM(av_spot_props[indexes_focal_plane, 3]*2.0), FWHM_x_err=convert_width_to_FHWM(std_spot_props[indexes_focal_plane, 3]*2.0), FWHM_y=convert_width_to_FHWM(av_spot_props[indexes_focal_plane, 4]*2.0), FWHM_y_err=convert_width_to_FHWM(std_spot_props[indexes_focal_plane, 4]*2.0), microns_per_pixel=microns_per_pixel, microns_per_pixel_err=microns_per_pixel_err)
peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_err=south_beam.calc_peak_intensity()
south_beam.a0, south_beam.a0_err=south_beam.calc_a0()
print(south_beam.a0)
print(south_beam.a0_err)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axs[0].scatter(f40_cam_positions, av_spot_props[:, 3]*2.0, color='r', label="x")
axs[0].vlines(f40_cam_positions, (av_spot_props[:, 3]-std_spot_props[:, 3])*2.0, (av_spot_props[:, 3]+std_spot_props[:, 3])*2.0, color='r', alpha=0.3)
axs[1].scatter(f40_cam_positions, av_spot_props[:, 4]*2.0, color='b', label="y")
axs[1].vlines(f40_cam_positions, (av_spot_props[:, 4]-std_spot_props[:, 4])*2.0, (av_spot_props[:, 4]+std_spot_props[:, 4])*2.0, color='b', alpha=0.3)
axs[0].scatter(f40_cam_positions, waist_x, color='r', marker="+", label="fit to waist x")
axs[0].vlines(f40_cam_positions, waist_x*(1.0-frac_err_popt1), waist_x*(1.0+frac_err_popt1), color='r', alpha=0.7)
axs[1].scatter(f40_cam_positions, waist_y, color='b', marker="+", label="fit to waist y")
axs[1].vlines(f40_cam_positions, waist_y*(1.0-frac_err_popt2), waist_y*(1.0+frac_err_popt2), color='b', alpha=0.7)
axs[1].set_xlabel("Camera position along focus (a.u.)")
axs[0].set_xlabel("Camera position along focus (a.u.)")
axs[1].set_ylabel("Laser waist ($\mu m$)")
axs[0].set_ylabel("Laser waist ($\mu m$)")
axs[1].legend(loc=0)
axs[0].legend(loc=2)
plt.tight_layout()
plt.show()



# i=0
#
# spot_props=np.array(south_beam.focal_spot.get_spot_properties_lst_sqrd_fit(f40_img[i])).reshape(-1, 8)
# spot_props[:,1:5]=spot_props[:,1:5]/microns_per_pixel
# print(spot_props)
#
# x_max=len(f40_img[0][0])#*pix_size_x/mag
# y_max=len(f40_img[0])
# x = np.linspace(0, x_max, x_max)#*microns_per_pixel
# y = np.linspace(0, y_max, y_max)#*microns_per_pixel
# X, Y = np.meshgrid(x, y)
#
# print('%s, %s, %s, %s, %s, %s, %s'%(spot_props[i,0], spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5], spot_props[i,6]))
# fitted_gauss=two_d_gaussian([X, Y], spot_props[i,0], spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5], spot_props[i,6]).reshape(X.shape)
# g=calc_ellipse(X, Y, spot_props[i,1], spot_props[i,2], spot_props[i,3], spot_props[i,4], spot_props[i,5])
# y_ell=Y[(g<1.1) & (g>0.99)]
# x_ell=X[(g<1.1) & (g>0.99)]
# # print(Y.shape)
# # print(y_ell.shape)
#
#
# # y_ell=y_ell[np.where(g>0.9)]
# # x_ell=x_ell[np.where(g>0.9)]
#
#
# fig, ax = plt.subplots(2)
# p1=ax[0].imshow(f40_img[i], vmin=np.amin(f40_img[i]), vmax=np.amax(f40_img[i]), origin='lower')
# ax[0].set_aspect('auto')
# ax[0].scatter(x_ell, y_ell, color='r', s=1.0)
# divider = make_axes_locatable(ax[0])
# cax1 = divider.append_axes('right', size='5%', pad=0.05)
# plt.colorbar(p1, cax=cax1)
# #plt.scatter(centre_x_pos[i], centre_y_pos[i])
# p2=ax[1].pcolor(X, Y, fitted_gauss, vmin=np.amin(f40_img[i]), vmax=np.amax(f40_img[i]))
# divider = make_axes_locatable(ax[1])
# cax2 = divider.append_axes('right', size='5%', pad=0.05)
# plt.colorbar(p2, cax=cax2)
# plt.show()
