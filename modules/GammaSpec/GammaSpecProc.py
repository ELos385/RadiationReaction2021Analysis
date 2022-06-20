#GammaSpecProc.py
import sys
sys.path.append('../../')
import numpy as np
import scipy.optimize as opt
import math
import matplotlib
import matplotlib.pyplot as plt
#import cv2
from scipy.special import kv, kn, expi
#import emcee
#import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from scipy.stats import norm
# from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from numpy.linalg import pinv
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.GPR_with_integrated_spectrum import *
from setup import *
from lib.general_tools import *

plt.style.use('/home/el1103292/GitHub/RadiationReaction2021Analysis/lib/thesis.mplstyle')


dict_diag_to_crystal_pos={
'CsIStackTop':'Crystal_pos_Top.mat',
'CsIStackSide':'Crystal_pos_Side.mat'}

diag_view_to_response_fn={
'CsIStackTop':'CsIEnergy_V_ProjZ_interp',
'CsIStackSide':'CsIEnergy_H_ProjZ_interp'}

diag_view_to_correction_factor={
'CsIStackTop':['Top_corr_factor_mean', 'Top_corr_factor_se'],
'CsIStackSide':['Side_corr_factor_mean', 'Side_corr_factor_se']}

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
    Egamma_MeV_interp_out=np.logspace(np.log10(min(Egamma_MeV_interp)), np.log10(max(Egamma_MeV_interp)), 250)#np.linspace(min(Egamma_MeV_interp), max(Egamma_MeV_interp), 250)##np.logspace(np.log10(min(Egamma_MeV_interp)), np.log10(max(Egamma_MeV_interp)), 250)#np.linspace(min(Egamma_MeV_interp), max(Egamma_MeV_interp), 200)##np.linspace(min(Egamma_MeV_interp), max(Egamma_MeV_interp), 150)
    print(np.log10(min(Egamma_MeV_interp)))
    print(np.log10(max(Egamma_MeV_interp)))
    CsIEnergy_ProjZ_interp=np.array(Edep_mat[diag_view_to_response_fn[diag]]).reshape(Egamma_MeV_interp.shape[0], -1)
    print(CsIEnergy_ProjZ_interp.shape)
    CsIEnergy_ProjZ_interp_out=np.zeros((len(Egamma_MeV_interp_out), len(CsIEnergy_ProjZ_interp[0])))
    for i in range(len(CsIEnergy_ProjZ_interp[0])):
        CsIEnergy_ProjZ_interp_out[:, i]=np.interp(Egamma_MeV_interp_out, Egamma_MeV_interp, CsIEnergy_ProjZ_interp[:, i])
    return Egamma_MeV_interp_out, CsIEnergy_ProjZ_interp_out

def load_correction_factor(diag):
    """
    Loads correction factors for CsI stack from mat file
    """
    path_2_correction_factor_mat="../../calib/GammaStack/GammaSpec_corr_factor_NEW_espec.mat"
    correction_factor=loadmat(path_2_correction_factor_mat)
    keys=diag_view_to_correction_factor[diag]
    corr_factor_mean=np.array(correction_factor[keys[0]]).reshape(-1)
    corr_factor_se=np.array(correction_factor[keys[1]]).reshape(-1)
    return corr_factor_mean, corr_factor_se

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
    mask[mask==1.0]=1000000.0
    mask[mask==False]=background_mask
    mask[mask==1000000.0]=filter_mask
    return xv, yv, mask

def get_mean_std_Ec_height_from_MC(params):
    Ec_mean=np.mean(params[:, 1])
    height_mean=np.mean(params[:, 2])
    offset_mean=np.mean(params[:, 3])
    Ec_std=np.std(params[:, 1])
    height_std=np.std(params[:, 2])
    offset_std=np.std(params[:, 3])
    return Ec_mean, height_mean, offset_mean, Ec_std, height_std, offset_std

def objective_fn(x_interp, x, y):
    return np.interp(x_interp, x, y)

def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
        #return fitted model and std
		return model.predict(X, return_std=True)

def acquisition(X, Xsamples, model):
    # get current estimates of true function
    yhat, _ = surrogate(model, X)
    # calculate max value of current estimate true function
    best = max(yhat)
    print(best)
    # retrieve estimated mean and std of true model for samples
    mu, std = surrogate(model, Xsamples)
    #not sure why this is neccessary
    mu = mu[:, 0]
    #print(mu[:, 1])
    # calculate the probability of improvement: note this is maximised if std is small, and if mu-best is large.
    # so we're picking models which have been a) well sampled, so low std, and/or a large difference between mean
    # predicted from sampled data and the best model/max of best model? Not sure: check what y_hat returns.
    # add small number to std to we don't get infs as we increase no samples.
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs

# optimize the acquisition function
def opt_acquisition(X, y, model):
	# draw N random samples from x
	Xsamples = np.random.uniform(min(X), max(X), 100)
	Xsamples = Xsamples.reshape(-1, 1)

	# run acquisition function; returns the probability that each sample will improve surrogate model
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores: i.e. points most likely to improve surrogate model
	ix = np.argmax(scores)
    #return samples most likely to improve surrogate model
	return Xsamples[ix, 0]

class GammaStack():
    """
    GammaStack class contains functions for raw image processing, and spectral fitting.
    Properties are the dimensions and positions of the crystals and the energy deposition
    per crystal.
    """
    def __init__(self, coordinates, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, kernel=None, debug=False):
        self.coords=coordinates
        self.N_crystals_X=N_crystals_X
        self.N_crystals_Y=N_crystals_Y
        self.N_crystals_X_cutoff=14
        self.pos_array_X=pos_array_X
        self.pos_array_Y=pos_array_Y
        self.crystal_size_XY_pxl=crystal_size_XY_pxl
        self.rot_deg=rot_deg
        x_shift=12
        y_shift=12
        self.gamma_stack_full_area_coords=np.array([min(self.pos_array_X)+x_shift, min(self.pos_array_Y)+y_shift, max(self.pos_array_X)+x_shift, min(self.pos_array_Y)+y_shift, max(self.pos_array_X)+x_shift, max(self.pos_array_Y)+y_shift, min(self.pos_array_X)+x_shift, max(self.pos_array_Y)+y_shift]).reshape(4, 2)
        self.Egamma_MeV_interp=Egamma_MeV_interp.astype(float)
        self.CsIEnergy_ProjZ_interp=CsIEnergy_ProjZ_interp[:, :self.N_crystals_X]
        self.corr_factor_mean=corr_factor_mean
        self.corr_factor_se=corr_factor_se
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
            
        im=plt.imshow(img_cp)#, vmin=None, vmax=maxx)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
        #im_ratio = img_cp.shape[0]/img_cp.shape[1]
        #cb=plt.colorbar(im,fraction=0.004*im_ratio, pad=0.04)
        #cb.set_label('Counts')
        ax = cb.ax
        cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
        plt.xlabel("y (pixels)")
        plt.ylabel("x (pixels)")
        plt.tight_layout()
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
            im=plt.imshow(Intensity_fit)
            #cb=plt.colorbar(im)
            im_ratio = img.shape[0]/img.shape[1]
            cb=plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
            #cb.set_label('Counts')
            ax = cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel("y (pixels)")
            plt.ylabel("x (pixels)")
            plt.tight_layout()
            plt.show()

        # Need to perform median (haircut) filter to deal with hard hits
        #w_filters_mean, w_filters_std=self.filter_pack.get_hard_hits(img.copy()[mask_dim_Y, mask_dim_X]+0.0, filter_labels)

        #img_norm=(img.copy()-np.mean(w_filters_mean))/(Intensity_fit-np.mean(w_filters_mean))
        img_norm=(img.copy())/(Intensity_fit)
        if self.debug==True:
            
            im=plt.imshow(img_norm[400:800, :])#vmin=0.99, vmax=1.01
            #plt.ylim(400, 800)
            im_ratio = img_norm[400:800, :].shape[0]/img_norm[400:800, :].shape[1]
            cb=plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
            #cb.set_label('Counts')
            ax = cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            cb.ax.ticklabel_format(style='scientific')
            plt.xlabel("y (pixels)")
            plt.ylabel("x (pixels)")
            
            plt.tight_layout()
            plt.savefig('/data/analysis/GEMINI/2021/App20110008-1/Results/GammaSpec/gamma_stack_normed_gradient_top.png')
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
            mask_p_img=mask_outside_filterpack+img[Y, X]
            im=plt.imshow(mask_p_img)
            im_ratio = mask_p_img.shape[0]/mask_p_img.shape[1]
            cb=plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04,  format='%.0e')
            ax = cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            #cb.ax.ticklabel_format(style='scientific')
            plt.xlabel("y (pixels)")
            plt.ylabel("x (pixels)")
            plt.tight_layout()
            plt.savefig('/data/analysis/GEMINI/2021/App20110008-1/Results/GammaSpec/gamma_stack_cropped_original_img_top.png')
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
            im=plt.imshow(data_normed_noise_crystal_only)
            im_ratio = data_normed_noise_crystal_only.shape[0]/data_normed_noise_crystal_only.shape[1]
            cb=plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
            ax = cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel("y (pixels)")
            plt.ylabel("x (pixels)")
            plt.tight_layout()
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
            im=plt.imshow(data_normed_noise_crystal_only_mean)
            im_ratio = data_normed_noise_crystal_only_mean.shape[0]/data_normed_noise_crystal_only_mean.shape[1]
            cb=plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04, format='%.0e')
            #cb.ax.ticklabel_format(style='scientific')
            #cb.set_label('Counts')
            ax = cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel("y (pixels)")
            plt.ylabel("x (pixels)")
            plt.tight_layout()
            plt.savefig('/data/analysis/GEMINI/2021/App20110008-1/Results/GammaSpec/final_gamma_stack_crystal_array_top.png')
            plt.show()
            
        data_normed_noise_crystal_only_summed=data_normed_noise_crystal_only_summed*self.corr_factor_mean
        data_normed_noise_crystal_only_summed_div=data_normed_noise_crystal_only_summed/np.max(data_normed_noise_crystal_only_summed)#np.mean(data_normed_noise_crystal_only_summed)
        return data_normed_noise_crystal_only_summed_div[0:self.N_crystals_X_cutoff]#-data_normed_noise_crystal_only_summed_div[0]

    def calc_Compton_energy_spec(self, Ec, height, offset):
        """
        Returns Compton spectrum given a ctitical energy
        """
        spec= height*(self.Egamma_MeV_interp/Ec)**-(2.0/3.0)*np.exp(-self.Egamma_MeV_interp/Ec)+offset
        return spec#/max(spec)

    def calc_Brems_energy_spec(self, Te_MeV, B):
        gff=3.0**0.5/np.pi*np.exp(self.Egamma_MeV_interp/(2.0*Te_MeV))*kn(0, self.Egamma_MeV_interp/(2.0*Te_MeV))#expi(0.5*(E/Te_MeV)**2)
        dP_dE_brehms=B*1.0/(Te_MeV)**0.5*np.exp(-self.Egamma_MeV_interp/Te_MeV)*gff
        # print("gff=%s"%gff)
        # print("B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)=%s"%(B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)))
        return dP_dE_brehms

    def calc_theoretical_Compton_signal_summed_over_columns(self, x, Ec, height, offset):#, height, offset):#theta, QE, filter_transmission, E_axis_keV, no_sync
        """
        Returns response of crystal array to a gamma beam with energy spectrum well-modelled by a Comptons
        spectrum.
        """
        Predicted_signal_summed_over_columns=np.zeros(self.N_crystals_X)
        dN_dE=self.calc_Compton_energy_spec(Ec, height, offset)
        for i in range(0, self.N_crystals_X):
            Predicted_signal_summed_over_columns[i]=np.trapz(dN_dE*self.CsIEnergy_ProjZ_interp[:, i], self.Egamma_MeV_interp)
        Predicted_signal_summed_over_columns_out=Predicted_signal_summed_over_columns/np.max(Predicted_signal_summed_over_columns)#/np.mean(Predicted_signal_summed_over_columns)
        return Predicted_signal_summed_over_columns_out[0:self.N_crystals_X_cutoff]#*height+offset#height*Predicted_signal_summed_over_columns_out+offset#-Predicted_signal_summed_over_columns_out[0])
    

    def least_sqrs(self, function, guess, measured_signal):
        """
        Fits Compton spectrum given measured counts per column of crystals.
        """
        popt, pcov= opt.curve_fit(function, xdata=None, ydata=measured_signal, p0=guess, bounds=(0.0, 10000000.0))
        if (len(measured_signal) > len(guess)) and pcov is not None:
            sigma = np.sqrt(np.diag(pcov))
            return popt, sigma
        else:
            return [None, None], [None, None]

#     def calc_theoretical_Brems_signal_summed_over_columns(self, x, Ec, height, offset):#theta, QE, filter_transmission, E_axis_keV, no_sync
#         """
#         Returns response of crystal array to a gamma beam with energy spectrum well-modelled by a Brems
#         spectrum.
#         """
#         Predicted_signal_summed_over_columns=np.zeros(self.N_crystals_X)
#         dN_dE=self.calc_Brems_energy_spec(Ec, height)
#         for i in range(0, self.N_crystals_X):
#             Predicted_signal_summed_over_columns[i]=np.trapz(dN_dE*self.CsIEnergy_ProjZ_interp[:, i], self.Egamma_MeV_interp)
#         Predicted_signal_summed_over_columns_out=Predicted_signal_summed_over_columns/np.mean(Predicted_signal_summed_over_columns)
#         return height*Predicted_signal_summed_over_columns_out+offset#-Predicted_signal_summed_over_columns_out[0])


#     def Bayes_mcmc(self, guess, no_walkers, no_steps, no_burn, no_dim, Y_measured):#mean_energy_diff, dN_dgamma_approx
#         sampler = emcee.EnsembleSampler(no_walkers, no_dim, self.log_posterior, args=[Y_measured])#mean_energy_diff, dN_dgamma_approx,
#         sampler.run_mcmc(guess, no_steps)
#         params= sampler.chain[:, no_burn:, :].reshape(-1, no_dim)
#         sampler.reset()
#         return params

#     def log_posterior(self, theta, Y_measured):#mean_energy_diff, dN_dgamma_approx,
#         return self.log_likelihood_signal(theta, Y_measured)+self.log_prior(theta)#, Y_measured, no_sync)#+log_likelihood_difference(theta, mean_energy_diff, dN_dgamma_approx, no_sync)

#     def log_prior(self, theta):#, Y_measured, no_sync):
#         Is_positive=True
#         for i in range(0, len(theta)):
#             if theta[i]<0.0 or theta[i]>2000.0 or np.isnan(theta[i])==True:
#                 Is_positive=False
#         if Is_positive==False:
#             return -np.inf
#         else:
#             return 0.0

#     def log_likelihood_signal(self, theta, Y_measured):
#         sigma = theta[0]
#         Y_model=self.calc_theoretical_signal_summed_over_columns(None, theta[1], theta[2], theta[3])#theta, QE, filter_transmission, E_int, no_sync
#         log_likelihood=-np.inf
#         is_positive=True
#         for i in range(0, len(theta)):
#             if theta[i]<0 or np.isnan(theta[i])==True:
#                 is_positive=False
#         if is_positive==True:
#             log_likelihood=np.sum(-0.5*(Y_measured-Y_model)**2/sigma**2-0.5*np.log(2*np.pi)-np.log(sigma))
#             if np.isnan(log_likelihood)==True:
#                 log_likelihood=-np.inf
#         return log_likelihood

#     def generate_estimates(self, params, err_array, percent_std, no_walkers, no_dim):
#         guess=np.zeros((no_walkers, no_dim))
#         guess[:, 0]=np.random.normal(err_array, err_array*percent_std, no_walkers)
#         for j in range(0, len(params)):
#             guess[:, j+1]=np.random.normal(params[j], params[j]*percent_std, no_walkers)#np.random.normal(Ec_guess[i-1], Ec_guess[i-1]*0.2, no_walkers)
#             # guess[:, i*2+2]=np.random.normal(height[i], height[i]*percent_std, no_walkers)#height[i]*percent_std
#         return guess

#     def plot_bayes_inferred_spec(self, params):
#         sigma_fit1=np.mean(params[:, 0])
#         print('sigma_fit1=%s'%sigma_fit1)
#         Ec_mean, height_mean, offset_mean, Ec_std, height_std, offset_std=get_mean_std_Ec_height_from_MC(params)
#         dN_dEdOmega_fit=np.zeros((len(params), len(self.Egamma_MeV_interp)))
#         for j in range(0, len(params)):
#         	dN_dEdOmega_fit[j, :]=self.calc_Compton_energy_spec(params[j, 1], params[j, 2], params[j, 3])

#         print(('Ec_mean=%s +/-%s')%(Ec_mean, Ec_std))
#         print(('height_mean=%s+/-%s')%(height_mean, height_std))
#         print(('offset_mean=%s+/-%s')%(offset_mean, offset_std))
#         std=2*dN_dEdOmega_fit.std(0)
#         mu=dN_dEdOmega_fit.mean(0)
#         plt.plot(self.Egamma_MeV_interp, mu, color='b', label='BI E$_{crit}$=%s'%(round(Ec_mean, 2)))
#         plt.fill_between(self.Egamma_MeV_interp, mu - std, mu + std, color='b', alpha=0.4)
#         plt.xlabel('Energy /MeV')
#         plt.ylabel('BI Compton spectrum, $dN/d\gamma$')
#         plt.legend(loc=0)
#         return

#     def plot_transmission_inferred(self, params, Y_measured_mean):#, Y_measured_std):

#         Ec_mean, height_mean, ofset_mean, Ec_std, height_std, offset_std=get_mean_std_Ec_height_from_MC(params)

#         Y_inferred=np.zeros((len(params), self.N_crystals_X))
#         for j in range(0, len(params)):
#             Y_inferred[j, :]=self.calc_theoretical_signal_summed_over_columns(None, params[j, 1], params[j, 2], params[j, 3])
#         Y_inferred_mean=np.mean(Y_inferred, axis=0)#.flatten()
#         Y_inferred_std=np.std(Y_inferred, axis=0)#.flatten()

#         filter_nos=np.linspace(1, self.N_crystals_X, self.N_crystals_X)

#         plt.plot(filter_nos, Y_inferred_mean, color='red', label='Ec=%s'%(round(Ec_mean, 2)))
#         plt.fill_between(filter_nos, Y_inferred_mean-Y_inferred_std, Y_inferred_mean+Y_inferred_std, color='red', alpha=0.5)
#         plt.scatter(filter_nos, Y_measured_mean, color='b', marker='.', label='data')
#         #plt.vlines(filter_nos, Y_measured_mean-Y_measured_std, Y_measured_mean+Y_measured_std, color='b', alpha=0.5)
#         plt.xlabel('Column number')
#         plt.ylabel('Normalised counts')
#         plt.legend(loc=0)
#         return

#     def bayesian_opt_spec(self, measured_signal_summed_over_columns):
#         model = GaussianProcessRegressorIntegratedY(self.CsIEnergy_ProjZ_interp)
#         #inv_CsIEnergy_ProjZ_interp=pinv(self.CsIEnergy_ProjZ_interp)
#         #dN_dE_test=np.matmul(measured_signal_summed_over_columns, inv_CsIEnergy_ProjZ_interp)
#         # fit the model
#         X=self.Egamma_MeV_interp.reshape(-1, 1)
#         y=measured_signal_summed_over_columns[0].reshape(-1, 1)
#         model.fit(X, y)
#         # print(self.Egamma_MeV_interp.shape)
#         # for i in range(1, len(measured_signal_summed_over_columns)-1):
#         #     print(i)
#         #     # x=self.Egamma_MeV_interp.reshape(-1, 1)
#         #     # actual=dN_dE_test[i].reshape(-1, 1)
#         #     x = opt_acquisition(X, y, model)
#         #     # 	# sample the point
#         #     actual = objective_fn(x, self.Egamma_MeV_interp, dN_dE_test[i])
#         #     est, _ = surrogate(model, [[x]])
#         #     X = np.vstack((X, [[x]]))
#         #     y = np.vstack((y, [[actual]]))
#         #     model.fit(X, y)
#         # y_final, std_final=surrogate(model, self.Egamma_MeV_interp.reshape(-1, 1))
#         y_final=surrogate(model, self.Egamma_MeV_interp.reshape(-1, 1))

#         # for i in range(100):
#         # 	# select the next point to sample
#         # 	x = opt_acquisition(X, y, model)
#         # 	# sample the point
#         # 	actual = objective_fn(x, noise_scale)
#         # 	# summarize the finding
#         # 	est, _ = surrogate(model, [[x]])
#         # 	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
#         # 	# add the data to the dataset
#         # 	X = np.vstack((X, [[x]]))
#         # 	y = np.vstack((y, [[actual]]))
#         # 	# update the model
#         # 	model.fit(X, y)

#         return y_final
