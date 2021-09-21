#GammaSpecProc.py


import sys
sys.path.append('../../')
import numpy as np
import scipy.optimize as opt
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat

from setup import *
from lib.general_tools import *


dict_diag_to_crystal_pos={
'CsIStackTop':'Crystal_pos_Top.mat',
'CsIStackSide':'Crystal_pos_Side.mat'}

diag_view_to_response_fn={
'CsIStackTop':'CsIEnergy_V_ProjZ_interp',
'CsIStackSide':'CsIEnergy_H_ProjZ_interp'}

def load_crystal_properties(diag):
    """
    Loads crystal sizes and positions from mat file
    """
    path_2_crystal_pos="../../calib/GammaStack/%s"%(dict_diag_to_crystal_pos[diag])
    crystal_pos=loadmat(path_2_crystal_pos)
    N_crystals_X=int(np.array(crystal_pos['N_crystals_X']).reshape(-1)[0])#-1
    N_crystals_Y=int(np.array(crystal_pos['N_crystals_Y']).reshape(-1)[0])#-1
    crystal_size_XY_pxl=int(np.array(crystal_pos['crystal_size_XY_pxl']).reshape(-1)[0])
    pos_array_X=np.array(crystal_pos['pos_array_X']).reshape(-1)
    pos_array_Y=np.array(crystal_pos['pos_array_Y']).reshape(-1)
    if diag=='CsIStackTop':
        rot_deg=np.array(crystal_pos['rot_deg']).reshape(-1)[0]
    else:
        rot_deg=0.0
    coords=np.array([pos_array_X, pos_array_Y, pos_array_X+crystal_size_XY_pxl, pos_array_Y, pos_array_X+crystal_size_XY_pxl, pos_array_Y+crystal_size_XY_pxl, pos_array_X, pos_array_Y+crystal_size_XY_pxl])
    return coords.T.reshape(-1,4,2), N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg

def load_Edep_info(diag):
    """
    Loads energy deppsition per crystal from mat file
    """
    path_2_Edep_mat="../../calib/GammaStack/Interp_Edep.mat"
    Edep_mat=loadmat(path_2_Edep_mat)
    Egamma_MeV_interp=np.array(Edep_mat['Egamma_MeV_interp']).reshape(-1)
    CsIEnergy_ProjZ_interp=np.array(Edep_mat[diag_view_to_response_fn[diag]]).reshape(Egamma_MeV_interp.shape[0], -1)
    return Egamma_MeV_interp, CsIEnergy_ProjZ_interp

def create_masked_element(coord_array, background_mask, filter_mask, add_x, add_y):
    """
    Given 4 co-ordinates and 2 values, a1 and a2, this sets the value of all pixels inside the coordinates
    to a1, and all pixels outside to a2.
    """
    coord_array_extended=coord_array+np.array([-add_x, -add_y, +add_x, -add_y, add_x, add_y, -add_x, +add_y]).reshape(4, 2)
    left, right= np.min(coord_array_extended, axis=0), np.max(coord_array_extended, axis=0)
    x = np.arange(max(0, min(math.ceil(left[0]), 1073)), max(0, min(math.floor(right[0]), 1073)))#np.arange(math.ceil(left[0]), math.floor(right[0])+1)
    y = np.arange(max(0, min(math.ceil(left[1]), 1073)), max(0, min(math.floor(right[1]), 1073)))#[::-1]#np.arange(math.ceil(left[1]), math.floor(right[1])+1)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
    path = matplotlib.path.Path(coord_array_extended)
    mask = path.contains_points(points).astype(float)
    mask.shape = xv.shape
    print(xv.shape)
    mask[mask==1.0]=2222.0
    mask[mask==False]=background_mask
    mask[mask==2222.0]=filter_mask
    return xv, yv, mask

class GammaStack():
    """
    GammaStack class contains functions for raw image processing, and spectral fitting.
    Properties are the dimensions and positions of the crystals and the energy deposition
    per crystal.
    """
    def __init__(self, coordinates, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, kernel=None, debug=False):
        self.coords=coordinates
        self.N_crystals_X=N_crystals_X
        self.N_crystals_Y=N_crystals_Y
        self.pos_array_X=pos_array_X
        self.pos_array_Y=pos_array_Y
        self.crystal_size_XY_pxl=crystal_size_XY_pxl
        self.rot_deg=rot_deg
        x_shift=12
        y_shift=12
        self.gamma_stack_full_area_coords=np.array([min(self.pos_array_X)+x_shift, min(self.pos_array_Y)+y_shift, max(self.pos_array_X)+x_shift, min(self.pos_array_Y)+y_shift, max(self.pos_array_X)+x_shift, max(self.pos_array_Y)+y_shift, min(self.pos_array_X)+x_shift, max(self.pos_array_Y)+y_shift]).reshape(4, 2)
        self.Egamma_MeV_interp=Egamma_MeV_interp
        self.CsIEnergy_ProjZ_interp=CsIEnergy_ProjZ_interp[:, :self.N_crystals_X]
        self.calib_image=None
        self.kernel=kernel
        self.debug=debug

    def subtract_bkg(self, img):
        """
        Performs image rotation (if neccessary), dark count subtraction and
        applies haircut median filter if specified (i.e. if self.kernel!=None)
        """
        img_rot=rotate(img, self.rot_deg)
        if self.calib_image is not None:
            return img_rot-self.calib_image

        if self.kernel is not None:
            img_med =  median_filter(img_rot, size=self.kernel)
            threshold=5.0*np.std(img_rot)#1.5*IQR+upper_quantile
            diff=abs(img_rot-img_med)
            img_rot[diff>threshold]=img_med[diff>threshold]
        return img_rot

    def plot_contours(self, img):
        """
        Plots image with the crystal positions overplotted.
        """
        img_cp=self.subtract_bkg(img.copy())
        brightness=1000
        maxx=(max(img_cp.flatten()))
        for i in range(0, len(self.coords)):
            cv2.polylines(img_cp, np.int32([self.coords[i]]), color=(brightness, brightness, brightness), thickness=2, isClosed=True)
            #cv2.putText(img_cp, "{}".format(np.arange(1, self.N_crystals_X*self.N_crystals_X)[i]), (int(self.coords[i][0][0])+0, int(self.coords[i][0][1])+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (brightness,brightness,brightness), 2)
        plt.imshow(img_cp)#, vmin=None, vmax=maxx)
        plt.colorbar()
        plt.show()
        return

    def normalise_beam_profile(self, img, mask, mask_dim_X, mask_dim_Y, get_total_counts, get_beam_pointing):
        """
        Fits a 4th order 2D polynomial to the background and normalises the fitted variation in intensity
        of the image.
        """
        background_no_crystals=img[mask_dim_Y, mask_dim_X]+mask
        background_no_crystals_del_nan, X_no_crystals_del_nan, Y_no_crystals_del_nan=background_no_crystals[~np.isnan(background_no_crystals)], mask_dim_X[~np.isnan(background_no_crystals)], mask_dim_Y[~np.isnan(background_no_crystals)]

        Intensity_fit, X_fit, Y_fit=calc_2D_4th_order_polyfit(X_no_crystals_del_nan, Y_no_crystals_del_nan, background_no_crystals_del_nan, [0, len(img[0])], [0, len(img)])

        if self.debug==True:
            plt.imshow(Intensity_fit)
            plt.colorbar()
            plt.show()

        # Need to perform median (haircut) filter to deal with hard hits
        #w_filters_mean, w_filters_std=self.filter_pack.get_hard_hits(img.copy()[mask_dim_Y, mask_dim_X]+0.0, filter_labels)

        #img_norm=(img.copy()-np.mean(w_filters_mean))/(Intensity_fit-np.mean(w_filters_mean))
        img_norm=(img.copy())/(Intensity_fit)
        if self.debug==True:
            plt.imshow(img_norm, vmin=0.99, vmax=1.01)
            plt.colorbar()
            plt.show()
        return img_norm

    def create_mask(self, img, background_mask, filter_mask, add_x, add_y):
        """
        Creates a mask for every filter in the GammaStack
        """
        mask_new= np.full(img.shape, background_mask)
        XV, YV=[], []
        for i in range(0, len(self.coords)):
            xv, yv, mask=create_masked_element(self.coords[i], background_mask, filter_mask, add_x, add_y)
            mask_new[yv, xv] = mask[:]
            XV.append(xv)
            YV.append(yv)
        return mask_new, XV, YV

    def get_measured_signal_summed_over_columns(self, img, get_total_counts=False, get_beam_pointing=False):
        """
        Returns the post-normalisation (i.e. background intensity variations normed out) total counts for each
        column of crystals.
        """
        img=self.subtract_bkg(img)

        filter_mask, _, _=self.create_mask(img, 0.0, np.nan, 4, 0)#15)

        X, Y, mask_outside_filterpack=create_masked_element(self.gamma_stack_full_area_coords, np.nan, 0.0, 5, 20)

        if self.debug==True:
            plt.imshow(mask_outside_filterpack+img[Y, X])
            plt.colorbar()
            plt.show()

        # normalise out beam intensity profile and subtract mean counts due to hard hits
        beam_data=self.normalise_beam_profile(img, mask_outside_filterpack+filter_mask[Y, X], X, Y, get_total_counts, get_beam_pointing)#, get_total_counts, get_beam_pointing)
        img_norm=beam_data.reshape(img.shape)

        # get mask
        background_mask, XV, YV=self.create_mask(img_norm, np.nan, 0.0, 0, 0)

        YV_r=np.array(YV).reshape(self.N_crystals_Y, self.N_crystals_X, self.crystal_size_XY_pxl, self.crystal_size_XY_pxl)
        XV_r=np.array(XV).reshape(self.N_crystals_Y, self.N_crystals_X, self.crystal_size_XY_pxl, self.crystal_size_XY_pxl)
        img_norm_r=img_norm[YV_r, XV_r]
        data_normed_noise_crystal_only_summed=np.sum(np.sum(np.sum(img_norm_r, axis=0), axis=1), axis=1)

        if self.debug==True:
            data_normed_noise_crystal_only=np.transpose(img_norm_r, (0, 2, 1, 3)).reshape(self.N_crystals_Y*self.crystal_size_XY_pxl, self.N_crystals_X*self.crystal_size_XY_pxl)
            plt.imshow(data_normed_noise_crystal_only)
            plt.colorbar()
            plt.show()

            data_normed_noise_crystal_only_mean=np.zeros(data_normed_noise_crystal_only.shape)
            signal_summed_over_columns=np.zeros(self.N_crystals_X)
            for j in range(self.N_crystals_X):
                total_counts_per_column=0
                for i in range(self.N_crystals_Y):
                    total_counts_per_crystal=np.sum(data_normed_noise_crystal_only[(self.crystal_size_XY_pxl)*i:(self.crystal_size_XY_pxl)*(i+1), self.crystal_size_XY_pxl*j:self.crystal_size_XY_pxl*(j+1)])
                    data_normed_noise_crystal_only_mean[(self.crystal_size_XY_pxl)*i:(self.crystal_size_XY_pxl)*(i+1), self.crystal_size_XY_pxl*j:self.crystal_size_XY_pxl*(j+1)]=total_counts_per_crystal
                    total_counts_per_column+=total_counts_per_crystal
                signal_summed_over_columns[j]=total_counts_per_column
            plt.imshow(data_normed_noise_crystal_only_mean)
            plt.colorbar()
            plt.show()

        return data_normed_noise_crystal_only_summed/max(data_normed_noise_crystal_only_summed)

    def calc_Brems_energy_spec(self, Ec):
        """
        Returns Brems spectrum given a ctitical energy
        """
        return self.Egamma_MeV_interp**-(2.0/3.0)*np.exp(-self.Egamma_MeV_interp/Ec)

    def calc_theoretical_signal_summed_over_columns(self, x, Ec):#theta, QE, filter_transmission, E_axis_keV, no_sync
        """
        Returns response of crystal array to a gamma beam with energy spectrum well-modelled by a Brems
        spectrum.
        """
    	Predicted_signal_summed_over_columns=np.zeros(self.N_crystals_X)
    	dN_dE=self.calc_Brems_energy_spec(Ec)
    	for i in range(0, self.N_crystals_X):
    		Predicted_signal_summed_over_columns[i]=np.trapz(dN_dE*self.CsIEnergy_ProjZ_interp[:, i], self.Egamma_MeV_interp)
    	return Predicted_signal_summed_over_columns/max(Predicted_signal_summed_over_columns)

    def least_sqrs_Ec_Brems(self, guess, measured_signal):
        """
        Fits brems spectrum given measured counts per column of crystals.
        """
        popt, pcov= opt.curve_fit(self.calc_theoretical_signal_summed_over_columns, xdata=None, ydata=measured_signal, p0=guess)# bounds=(0.01, 100.0),
        if (len(measured_signal) > len(guess)) and pcov is not None:
            sigma = np.sqrt(np.diag(pcov))
            return popt, sigma
        else:
            return [None, None], [None, None]
