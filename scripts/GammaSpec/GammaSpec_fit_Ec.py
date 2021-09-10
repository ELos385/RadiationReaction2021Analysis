#GammaSpec_fit_Ec.py


import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from skimage.io import imread

from setup import *
from lib.general_tools import *
from lib.pipeline import *


# QE_dict={"Gematron":"Andor Cameras/QE/QE_Andor_ikon_HF_ID.txt", "Lundatron":"Andor Cameras/QE/QE_Andor_iKon_L_DD.txt"}

# def get_filter_coords(filter_coord_ID):
#     path="../../modules/On_axis_Xrays/filter_pack_resources/%s_filterpack_squares.txt"%(filter_coord_ID)
#     filter_coords=np.genfromtxt(path, delimiter=',', skip_header=1)
#     filter_pack_nos=filter_coords[:, 0]
#     filter_coords=filter_coords[:, 1:].reshape((-1,4,2))
#     # coords=np.array(coords, np.int32).reshape((-1,4,2))
#     return filter_pack_nos, filter_coords
#
# def get_QE(diag_name):
#     #use data loader instead?
#     path_QE=os.path.join("../../modules/On_axis_Xrays/", QE_dict[diag_name])
#     Andor_data=np.genfromtxt(path_QE, delimiter=',', skip_header=1)
#     return interp1d(Andor_data[:, 0], Andor_data[:, 1], fill_value="extrapolate")

def create_masked_element(coord_array, background_mask, filter_mask, add_x, add_y):
    #coord_array_extended=coord_array+np.array([+add_x, -add_y, +add_x, +add_y, -add_x, add_y, -add_x, -add_y]).reshape(4, 2)#np.outer(np.full(len(coord_array), 1.0),
    coord_array_extended=coord_array+np.array([-add_x, -add_y, +add_x, -add_y, add_x, add_y, -add_x, +add_y]).reshape(4, 2)
    left, right= np.min(coord_array_extended, axis=0), np.max(coord_array_extended, axis=0)
    print('left=%s'%left)
    print('right=%s'%right)
    print(left[0])
    print(math.ceil(left[0]))
    x = np.arange(max(0, min(math.ceil(left[0]), 1073)), max(0, min(math.floor(right[0])+1, 1073)))#np.arange(math.ceil(left[0]), math.floor(right[0])+1)
    y = np.arange(max(0, min(math.ceil(left[1]), 1073)), max(0, min(math.floor(right[1])+1, 1073)))#[::-1]#np.arange(math.ceil(left[1]), math.floor(right[1])+1)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
    path = matplotlib.path.Path(coord_array_extended)
    mask = path.contains_points(points).astype(float)
    mask.shape = xv.shape
    mask[mask==1.0]=2222.0
    mask[mask==False]=background_mask
    mask[mask==2222.0]=filter_mask
    return xv, yv, mask

dict_diag_to_crystal_pos={
'CsIStackTop':'Crystal_pos_Top.mat',
'CsIStackSide':'Crystal_pos_Side.mat'}

class GammaStack():
    def __init__(self, coordinates, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y):
        self.coords=coordinates
        self.N_crystals_X=N_crystals_X
        self.N_crystals_Y=N_crystals_Y
        self.pos_array_X=pos_array_X
        self.pos_array_Y=pos_array_Y
        self.gamma_stack_full_area_coords=np.array([min(self.pos_array_X), min(self.pos_array_Y), max(self.pos_array_X), min(self.pos_array_Y), max(self.pos_array_X), max(self.pos_array_Y), min(self.pos_array_X), max(self.pos_array_Y)]).reshape(4, 2)

    def plot_contours(self, img):
        #img_cp=self.subtract_bkg(img.copy())
        img_cp=img.copy()
        brightness=15000
        maxx=(max(img_cp.flatten()))
        for i in range(0, len(self.coords)):
            cv2.polylines(img_cp, np.int32([self.coords[i]]), color=(brightness, brightness, brightness), thickness=2, isClosed=True)
            #cv2.putText(img_cp, "{} {}".format(self.filter_labels.flatten()[i], self.filter_pack.filter_keys[self.filter_labels.flatten()[i]-1]), (int(self.filter_coords[i][0][0])-100, int(self.filter_coords[i][0][1]) + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (brightness,brightness,brightness), 2)
        plt.imshow(img_cp, vmin=None, vmax=maxx)
        plt.colorbar()
        plt.show()
        return

    def normalise_beam_profile(self, img, mask, mask_dim_X, mask_dim_Y, get_total_counts, get_beam_pointing):
        background_no_crystals=img[mask_dim_Y, mask_dim_X]+mask
        background_no_crystals_del_nan, X_no_crystals_del_nan, Y_no_crystals_del_nan=background_no_crystals[~np.isnan(background_no_crystals)], mask_dim_X[~np.isnan(background_no_crystals)], mask_dim_Y[~np.isnan(background_no_crystals)]

        Intensity_fit, X_fit, Y_fit=calc_2D_4th_order_polyfit(X_no_crystals_del_nan, Y_no_crystals_del_nan, background_no_crystals_del_nan, [0, len(img[0])], [0, len(img)])
        plt.imshow(background_no_crystals)
        # plt.imshow(Intensity_fit)
        plt.colorbar()
        plt.show()

        plt.imshow(Intensity_fit)
        plt.colorbar()
        plt.show()

        # Need to perform median (haircut) filter to deal with hard hits
        #w_filters_mean, w_filters_std=self.filter_pack.get_hard_hits(img.copy()[mask_dim_Y, mask_dim_X]+0.0, filter_labels)

        #img_norm=(img.copy()-np.mean(w_filters_mean))/(Intensity_fit-np.mean(w_filters_mean))
        img_norm=(img.copy())/(Intensity_fit)
        plt.imshow(img_norm, vmin=0.99, vmax=1.01)
        plt.colorbar()
        plt.show()
        arr_out=[img_norm]
        # if get_total_counts==True:
        #     Total_counts=np.sum(Intensity_fit-np.mean(w_filters_mean))
        #     arr_out.append(Total_counts)
        #
        # if get_beam_pointing==True:
        #     centre=[len(img[0])/2.0, len(img)/2.0]#x, y
        #     x_position_of_max=X_fit[np.where(Intensity_fit==np.amax(Intensity_fit))]#numpy.where(arr2D == numpy.amax(arr2D))
        #     y_position_of_max=Y_fit[np.where(Intensity_fit==np.amax(Intensity_fit))]
        #     arr_out.append([x_position_of_max, y_position_of_max])
        #     # radial_disp_from_centre=((centre[0]*self.pixel_size[0]-x_position_of_max*self.pixel_size[0])**2+(centre[1]*self.pixel_size[1]-y_position_of_max*self.pixel_size[1])**2)**0.5
        #     # theta=np.arctan((y_position_of_max-centre[1])/(x_position_of_max-centre[0]))
        #     # arr_out.append([radial_disp_from_centre, theta])

        return arr_out

    def create_mask(self, img, background_mask, filter_mask, add_x, add_y):
        temp=10.0
        mask_new, labels, unique_labels = np.full(img.shape, background_mask), np.full(img.shape, background_mask), np.full(img.shape, background_mask)
        for i in range(0, len(self.coords)):#len(self.filter_coords)-1
            xv, yv, mask=create_masked_element(self.coords[i], background_mask, filter_mask, add_x, add_y)
            mask_new[yv, xv] = mask[:]
            # labels[yv, xv] = self.filter_labels.flatten()[i]
            # unique_labels[yv, xv] = i-1
        # matplotlib.pyplot.imshow(mask_new)
        # matplotlib.pyplot.colorbar()
        # matplotlib.pyplot.show()
        return mask_new#, labels, unique_labels

    def get_measured_filter_transmission(self, img, get_total_counts=False, get_beam_pointing=False):
        # img=self.subtract_bkg(img)

        # get mask
        # background_mask, filter_labels, filter_unique_labels=self.create_mask(img, np.nan, 0.0, 0, 0)
        background_mask=self.create_mask(img, np.nan, 0.0, 0, 0)

        plt.imshow(background_mask)
        plt.show()

        # filter_mask, _, _ =self.create_mask(img, 0.0, np.nan, 0, 0)#15)
        filter_mask =self.create_mask(img, 0.0, np.nan, 5, 0)#15)

        plt.imshow(filter_mask)
        plt.show()

        X, Y, mask_outside_filterpack=create_masked_element(self.gamma_stack_full_area_coords, np.nan, 0.0, 10, 10)

        # normalise out beam intensity profile and subtract mean counts due to hard hits
        beam_data=self.normalise_beam_profile(img, mask_outside_filterpack+filter_mask[Y, X], X, Y, get_total_counts, get_beam_pointing)#, get_total_counts, get_beam_pointing)
        img_norm=beam_data[0]
        data_normed_noise_sub_no_background=img_norm+background_mask

        self.plot_contours(data_normed_noise_sub_no_background)
        plt.imshow(img_norm)#, vmin=0.0, vmax=0.85
        # plt.imshow(image_highres, vmin=0.95, vmax=1.05)
        plt.colorbar()
        plt.show()

        plt.imshow(data_normed_noise_sub_no_background.copy(),  vmin=0.0, vmax=1.01)
        plt.colorbar()
        plt.plot()
        # Y_measured_mean, Y_measured_std, Y_filter_indx=[], [], []
        # for i in range(0, int(self.N_crystals_Y*self.N_crystals_X)):
        #
        #     unique_mask=data_normed_noise_sub_no_background.copy()
        #     #signal_same_filters=unique_mask[filter_labels==self.filter_pack.filter_no[i]]
        #
        #     #remove outliers:
        #     threshold=np.nanmean(signal_same_filters)+70.0*np.nanstd(signal_same_filters)
        #     signal_same_filters_threshd = signal_same_filters[signal_same_filters < threshold]

            # plt.plot(signal_same_filters[~np.isnan(signal_same_filters)], color="b", alpha=0.4)
            # plt.plot(signal_same_filters_threshd, color="r", alpha=0.4)
            # plt.show()

        # Y_measured_mean.append(np.nanmean(signal_same_filters_threshd))
        # Y_measured_std.append(np.nanstd(signal_same_filters_threshd))
        # arr_out=[np.array(Y_measured_mean), np.array(Y_measured_std)]
        # if get_total_counts==True:
        #     arr_out.append(beam_data[1])
        #
        # if get_beam_pointing==True:
        #     arr_out.append(beam_data[2])
        return #arr_out


diagT='CsIStackTop'
# diagS='CsIStackTop'
date='20210608'
run='run10'
shot='Shot029'

pathT=ROOT_DATA_FOLDER+diagT+'/'+date+'/'+run+'/'+shot+'.tif'
path_2_crystal_pos="../../calib/GammaStack/%s"%(dict_diag_to_crystal_pos[diagT])
# path_2_crystal_pos="../../calib/GammaStack/%s"%(dict_diag_to_crystal_pos[diagS])
path_2_Edep_mat="../../calib/GammaStack/Interp_Edep.mat"
# GammaSide_Img=imread()
GammaTop_Img=imread(pathT)

crystal_pos_Top=loadmat(path_2_crystal_pos)
N_crystals_X=np.array(crystal_pos_Top['N_crystals_X']).reshape(-1)#-1
N_crystals_Y=np.array(crystal_pos_Top['N_crystals_Y']).reshape(-1)#-1
crystal_size_XY_pxl=np.array(crystal_pos_Top['crystal_size_XY_pxl']).reshape(-1)
pos_array_X=np.array(crystal_pos_Top['pos_array_X']).reshape(-1)
pos_array_Y=np.array(crystal_pos_Top['pos_array_Y']).reshape(-1)
rot_deg=np.array(crystal_pos_Top['rot_deg']).reshape(-1)

print(N_crystals_X)

Edep_mat=loadmat(path_2_Edep_mat)
Egamma_MeV_interp=np.array(Edep_mat['Egamma_MeV_interp']).reshape(-1)
CsIEnergy_H_ProjZ_interp=np.array(Edep_mat['CsIEnergy_H_ProjZ_interp']).reshape(-1)
CsIEnergy_V_ProjZ_interp=np.array(Edep_mat['CsIEnergy_V_ProjZ_interp']).reshape(-1)
coords=np.array([pos_array_X, pos_array_Y, pos_array_X+crystal_size_XY_pxl, pos_array_Y, pos_array_X+crystal_size_XY_pxl, pos_array_Y+crystal_size_XY_pxl, pos_array_X, pos_array_Y+crystal_size_XY_pxl])

CsIStackTop=GammaStack(coords.T.reshape(-1,4,2), N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y)

print(coords)
# print(pos_array_X)
# print(pos_array_Y)
# print(pos_array_X+crystal_size_XY_pxl)
# print(coords.shape)
# coords=coords.T.reshape(-1,4,2)#.reshape(-1,4,2)
# print(coords)
img_rot=rotate(GammaTop_Img, rot_deg[0])
print(img_rot.shape)
CsIStackTop.plot_contours(img_rot)


CsIStackTop.get_measured_filter_transmission(img_rot, get_total_counts=False, get_beam_pointing=False)



# plt.imshow(GammaTop_Img)
# plt.show()
