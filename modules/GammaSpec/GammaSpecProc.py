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

from lib.GPR_with_integrated_spectrum import *
from setup import *
from lib.general_tools import *
from modules.GammaSpec.Gamma_stack_ref import *


dict_diag_to_crystal_pos={
'CsIStackTop':'Crystal_pos_Top.mat',
'CsIStackSide':'Crystal_pos_Side.mat'}

diag_view_to_response_fn={
'CsIStackTop':'CsIEnergy_V_ProjZ_interp',
'CsIStackSide':'CsIEnergy_H_ProjZ_interp'}

# diag_view_to_correction_factor={
# 'CsIStackTop':['Top_corr_factor_mean', 'Top_corr_factor_se'],
# 'CsIStackSide':['Side_corr_factor_mean', 'Side_corr_factor_se']}

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
    CsIEnergy_ProjZ_interp=np.array(Edep_mat[diag_view_to_response_fn[diag]]).reshape(Egamma_MeV_interp.shape[0], -1)
    CsIEnergy_ProjZ_interp_out=np.zeros((len(Egamma_MeV_interp_out), len(CsIEnergy_ProjZ_interp[0])))
    for i in range(len(CsIEnergy_ProjZ_interp[0])):
        CsIEnergy_ProjZ_interp_out[:, i]=np.interp(Egamma_MeV_interp_out, Egamma_MeV_interp, CsIEnergy_ProjZ_interp[:, i])
    return Egamma_MeV_interp_out, CsIEnergy_ProjZ_interp_out

def load_correction_factor(diag):
    """
    Loads correction factors for CsI stack from mat file
    """
    path_2_correction_factor_mat="../../calib/GammaStack/gamma_stack_correction_factor_EL.pkl"
    correction_factor=load_object(path_2_correction_factor_mat)
    keys=list(correction_factor[diag].keys())
    #print(correction_factor[diag])
    corr_factor_mean=np.array(correction_factor[diag][keys[0]]).reshape(-1)
    corr_factor_se=np.array(correction_factor[diag][keys[1]]).reshape(-1)
    return corr_factor_mean, corr_factor_se

def initialise_gamma_stack_obj(diag, date):
    #load crystal properties from mat file
    coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
    #load energy deposition info from mat file
    Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)
    #load correction factor for CsI stack
    corr_factor_mean, corr_factor_se=load_correction_factor(diag)
    # load image for background subtraction
    path_calib=ROOT_DATA_FOLDER+diag+'/'+date+'/'+bkg_img_dict[date][0]+'/%s'%(bkg_img_dict[date][1])
    calib_img=imread(path_calib)

    CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, hard_hits_filter=None, calib_img=calib_img, kernel=5, debug=False)
    return CsIStack

def create_masked_element(img, coord_array, background_mask, filter_mask, add_x, add_y):
    """
    Given 4 co-ordinates and 2 values, a1 and a2, this sets the value of all pixels inside the coordinates
    to a1, and all pixels outside to a2.
    """
    # add_x=0
    # add_y=100
    b=1000
    #coord_array_extended=coord_array+np.array([-add_x, -add_y, +add_x, -add_y, add_x, add_y, -add_x, +add_y]).reshape(4, 2)
    coord_array_extended=coord_array+np.array([-add_x, -add_y, +add_x, -add_y, add_x, add_y, -add_x, +add_y]).reshape(4, 2)
    # plt.scatter(coord_array_extended[:, 0], coord_array_extended[:, 1], color='black')
    # plt.scatter(coord_array[0, 0], coord_array[0, 1], color='b')
    # plt.scatter(coord_array[1, 0], coord_array[1, 1], color='r')
    # plt.scatter(coord_array[2, 0], coord_array[2, 1], color='g')
    # plt.scatter(coord_array[3, 0], coord_array[3, 1], color='orange')

    #cv2.polylines(img, np.int32([coord_array]), color=(b, b, b), thickness=1, isClosed=True)
    #cv2.polylines(img, np.int32([coord_array_extended]), color=(b, b, b), thickness=1, isClosed=True)

    left, right= np.min(coord_array_extended, axis=0), np.max(coord_array_extended, axis=0)
    x = np.arange(max(0, min(math.ceil(left[0]), 1073)), max(0, min(math.floor(right[0]), 1073)))#np.arange(math.ceil(left[0]), math.floor(right[0])+1)
    y = np.arange(max(0, min(math.ceil(left[1]), 1073)), max(0, min(math.floor(right[1]), 1073)))#[::-1]#np.arange(math.ceil(left[1]), math.floor(right[1])+1)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
    path = matplotlib.path.Path(coord_array_extended)
    mask = path.contains_points(points).astype(float)
    mask.shape = xv.shape
    mask[mask==1.0]=1000000.0
    mask[mask==0.0]=background_mask
    mask[mask==1000000.0]=filter_mask
    xv[xv>=img.shape[1]]=img.shape[1]-1
    yv[yv>=img.shape[0]]=img.shape[0]-1
    return xv, yv, mask

def get_mean_std_Ec_height_from_MC(params, norm_factor):
    Ec_mean=np.mean(params[:, 1])
    height_mean=np.mean(params[:, 2])
    Ec_std=np.std(params[:, 1])
    height_std=np.std(params[:, 2])
    scale_mean=np.mean(params[:, 3])
    scale_std=np.std(params[:, 3])
    arr_out=[Ec_mean/norm_factor, Ec_std/norm_factor, height_mean/norm_factor, height_std/norm_factor, scale_mean/norm_factor, scale_std/norm_factor]
    return arr_out#Ec_mean/norm_factor, Ec_std/norm_factor, height_mean/norm_factor, height_std/norm_factor#, offset_mean/norm_factor, offset_std/norm_factor

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

def apply_haircut_filter(img_rot, kernel):
    img_med =  median_filter(img_rot, size=kernel)
    threshold=5.0*np.std(img_rot)#1.5*IQR+upper_quantile
    diff=abs(img_rot-img_med)
    img_rot[diff>threshold]=img_med[diff>threshold]
    return img_rot

def remove_hard_hits(img_sub, hard_hits_filter):
    blob_size=15
    rep=10000
    for i in range(0, hard_hits_filter):
        img_med=median_filter(img_sub, size=3+i*2)
        img_diff = abs(img_med-img_sub)
        threshold=np.std(img_sub)*5.0#+np.mean(img_sub)
        mask=create_mask(img_diff, blob_size, threshold, rep)
        img_sub[mask==rep]=img_med[mask==rep]
    return img_sub

def get_norm_factor():
    norm_factor_file=CALIB_FOLDER+"/GammaStack/gamma_stack_nomalisation_factor.pkl"
    norm_factor_data=load_object(norm_factor_file)
    return norm_factor_data[list(norm_factor_data.keys())[0]]

def Bayes_mcmc(guess, no_walkers, no_steps, no_burn, no_dim, Y_measured_top, Y_measured_side, CsITop, CsISide, pool):#mean_energy_diff, dN_dgamma_approx
    # sampler = emcee.EnsembleSampler(no_walkers, no_dim, log_posterior, args=[Y_measured_top, Y_measured_side, CsITop, CsISide])#mean_energy_diff, dN_dgamma_approx,
    # sampler.run_mcmc(guess, no_steps)
    # params= sampler.chain[:, no_burn:, :].reshape(-1, no_dim)
    # sampler.reset()
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(no_steps)

    # This will be useful to testing convergence
    old_tau = -np.inf

    sampler = emcee.EnsembleSampler(no_walkers, no_dim, log_posterior, args=[Y_measured_top, Y_measured_side, CsITop, CsISide], pool=pool)
    #sampler.run_mcmc(initial_estimate, no_steps)
            # Now we'll sample for up to no_steps steps
    for sample in sampler.sample(guess, iterations=no_steps):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    params=sampler.chain[:, no_burn:, :].reshape(-1, no_dim)
    sampler.reset()
    return params

def log_posterior(theta, Y_measured_top, Y_measured_side, CsITop, CsISide):#mean_energy_diff, dN_dgamma_approx,
    return CsITop.log_likelihood_signal(theta, Y_measured_top, Invert=True)+CsISide.log_likelihood_signal(theta, Y_measured_side, Invert=False)+log_prior(theta)#, Y_measured, no_sync)#+log_likelihood_difference(theta, mean_energy_diff, dN_dgamma_approx, no_sync)

def log_prior(theta):#, Y_measured, no_sync):
    Is_positive=True
    for i in range(0, len(theta)):
        if theta[i]<0.0 or theta[i]>2000.0 or np.isnan(theta[i])==True:
            Is_positive=False
    if Is_positive==False:
        return -np.inf
    else:
        return 0.0

class GammaStack():
    """
    GammaStack class contains functions for raw image processing, and spectral fitting.
    Properties are the dimensions and positions of the crystals and the energy deposition
    per crystal.
    """
    def __init__(self, coordinates, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, hard_hits_filter=None, calib_img=None, kernel=None, debug=False):
        self.coords=coordinates
        self.N_crystals_X=N_crystals_X
        self.N_crystals_X_cuttoff=14
        self.N_crystals_Y=N_crystals_Y
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
        self.calib_image=calib_img
        self.kernel=kernel
        self.debug=debug
        self.hard_hits_filter=hard_hits_filter
        norm_factor=get_norm_factor()
        self.norm_factor=norm_factor

    def subtract_bkg(self, img):
        """
        Performs image rotation (if neccessary), dark count subtraction and
        applies haircut median filter if specified (i.e. if self.kernel!=None)
        """
        if self.calib_image is not None:
            img = cv2.subtract(img, self.calib_image)

        img_rot=rotate(img, self.rot_deg)
        img_rot=cv2.subtract(img_rot, np.median(img))
        if self.kernel is not None:
            img_rot=apply_haircut_filter(img_rot, self.kernel)
            # img_med =  median_filter(img_rot, size=self.kernel)
            # threshold=5.0*np.std(img_rot)#1.5*IQR+upper_quantile
            # diff=abs(img_rot-img_med)
            # img_rot[diff>threshold]=img_med[diff>threshold]
        if self.hard_hits_filter is not None:
            print()
            img_rot=remove_hard_hits(img_rot.copy(), self.hard_hits_filter)
        return img_rot

    def plot_contours(self, img):
        """
        Plots image with the crystal positions overplotted.
        """
        #img_cp=self.subtract_bkg(img.copy())
        img_cp=img.copy()
        maxx=(max(img_cp.flatten()))
        brightness=100#maxx
        for i in range(0, len(self.coords)):
            cv2.polylines(img_cp, np.int32([self.coords[i]]), color=(brightness, brightness, brightness), thickness=2, isClosed=True)
            #cv2.putText(img_cp, "{}".format(np.arange(1, self.N_crystals_X*self.N_crystals_X)[i]), (int(self.coords[i][0][0])+0, int(self.coords[i][0][1])+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (brightness,brightness,brightness), 2)
        img_cp=img_cp[450:750, 200:]#[150:450, 100:900]
        im=plt.imshow(img_cp, vmax=brightness)#, vmin=None, vmax=maxx)
        im_ratio=img_cp.shape[0]/img_cp.shape[1]
        cb=plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04)
        ax=cb.ax
        cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
        plt.xlabel('y (pixels)')
        plt.ylabel('x (pixels)')
        plt.tight_layout()
        plt.savefig('gamma_stack_side_filters_encircled.png')
        plt.show()
        return

    def normalise_beam_profile(self, img, mask, mask_dim_X, mask_dim_Y, get_total_counts, get_beam_pointing):
        """
        Fits a 4th order 2D polynomial to the background and normalises the fitted variation in intensity
        of the image.
        """
        background_no_crystals=img[mask_dim_Y, mask_dim_X]+mask

        background_no_crystals=remove_hard_hits(background_no_crystals.copy(), 5)

        #w_filters_mean, w_filters_std=self.filter_pack.get_hard_hits(background_no_crystals.copy()[mask_dim_Y, mask_dim_X]+0.0, filter_labels)
        #background_no_crystals=apply_haircut_filter(background_no_crystals, 5)
        if self.debug==True:
            plt.imshow(background_no_crystals, vmin=0, vmax=10)
            plt.title("background_no_crystals")
            plt.colorbar()
            plt.show()

        background_no_crystals_del_nan, X_no_crystals_del_nan, Y_no_crystals_del_nan=background_no_crystals[~np.isnan(background_no_crystals)], mask_dim_X[~np.isnan(background_no_crystals)], mask_dim_Y[~np.isnan(background_no_crystals)]

        # if self.debug==True:
        #     print("X_no_crystals_del_nan.shape=%s"%X_no_crystals_del_nan.shape)
        #     plt.pcolor(X_no_crystals_del_nan, Y_no_crystals_del_nan, background_no_crystals_del_nan)
        #     plt.title("masked crystals: background only")
        #     plt.colorbar()
        #     plt.show()

        Intensity_fit, X_fit, Y_fit=calc_2D_4th_order_polyfit(X_no_crystals_del_nan, Y_no_crystals_del_nan, background_no_crystals_del_nan, [0, len(img[0])], [0, len(img)])

        if self.debug==True:
            plt.imshow(Intensity_fit)#, vmin=0, vmax=100)
            plt.title("Intensity fit to background")
            plt.colorbar()
            plt.show()

        # Need to perform median (haircut) filter to deal with hard hits

        #w_filters_mean, w_filters_std=self.filter_pack.get_hard_hits(img.copy()[mask_dim_Y, mask_dim_X]+0.0, filter_labels)

        #img_norm=(img.copy()-np.mean(w_filters_mean))/(Intensity_fit-np.mean(w_filters_mean))
        #img_norm=(img.copy())/(Intensity_fit)

        img_norm=img_rot=cv2.subtract(img.copy(), Intensity_fit,dtype=cv2.CV_64F)
        if self.debug==True:
            plt.imshow(img_norm, vmin=0.99, vmax=1.01)
            plt.title("Intensity fit to background restricted limits")
            plt.colorbar()
            plt.show()

            plt.imshow(img_norm, vmin=0, vmax=100.0)
            plt.title("Intensity fit to background no limits")
            plt.colorbar()
            plt.show()
        return img_norm

    def create_mask(self, img, background_mask, filter_mask, add_x, add_y):
        """
        Creates a mask for every filter in the GammaStack
        """
        mask_new= np.full(img.shape, background_mask)
        XV, YV=[], []
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # im=ax.imshow(img)#, vmin=-1, vmax=1)
        for i in range(0, len(self.coords)):
            xv, yv, mask=create_masked_element(img, self.coords[i], background_mask, filter_mask, add_x, add_y)
            # xv=xv[0:img.shape[0], 0:img.shape[1]]
            # yv=yv[0:img.shape[0], 0:img.shape[1]]
            # mask=mask[0:img.shape[0], 0:img.shape[1]]


            mask_new[yv, xv] += mask[:]
            XV.append(xv)
            YV.append(yv)
        #     ax.pcolor(xv, yv, mask),# vmin=-1, vmax=1)
        # plt.title("mask creation")
        # plt.colorbar(im)
        # plt.show()
        #mask_new[mask_new>92.0]=background_mask

        return mask_new, XV, YV

    def get_measured_signal_rows_and_cols(self, img, get_total_counts=False, get_beam_pointing=False):
        """
        Returns the post-normalisation (i.e. background intensity variations normed out) total counts for each
        crystal.
        """

        if self.debug==True:
            img2plt=img[450:750, 200:]
            im=plt.imshow(img2plt, vmax=200)#, vmin=0, vmax=100)
            im_ratio=img2plt.shape[0]/img2plt.shape[1]
            cb=plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04)
            ax=cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel('y (pixels)')
            plt.ylabel('x (pixels)')
            plt.tight_layout()
            plt.savefig('gamma_stack_cropped_original_img_side.png')
            plt.show()

        img=self.subtract_bkg(img)
        #was 5, 0
        filter_mask, _, _=self.create_mask(img, 0.0, np.nan, 7, 20)#15)

        if self.debug==True:
            im=plt.imshow(filter_mask+img, vmin=0, vmax=100)
            plt.colorbar(im)
            plt.show()

        X, Y, mask_outside_filterpack=create_masked_element(img, self.gamma_stack_full_area_coords, np.nan, 0.0, 20, 60)

        if self.debug==True:
            plt.imshow(mask_outside_filterpack+img[Y, X], vmin=0, vmax=100)
            plt.colorbar()
            plt.show()

        # normalise out beam intensity profile and subtract mean counts due to hard hits

        beam_data=self.normalise_beam_profile(img, mask_outside_filterpack+filter_mask[Y, X], X, Y, get_total_counts, get_beam_pointing)#, get_total_counts, get_beam_pointing)
        img_norm=beam_data.reshape(img.shape)

        if self.debug==True:
            filter_masked_img=img_norm[450:750, 200:]#+filter_mask
            im=plt.imshow(filter_masked_img)#, vmin=0, vmax=10)
            #plt.title("intensity normed img")
            im_ratio=filter_masked_img.shape[0]/filter_masked_img.shape[1]
            cb=plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04)
            ax=cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel('y (pixels)')
            plt.ylabel('x (pixels)')
            plt.tight_layout()
            plt.savefig('gamma_stack_normed_gradient_side.png')
            plt.show()

        #self.plot_contours(img_norm)

        # get mask
        background_mask, XV, YV=self.create_mask(img_norm, np.nan, 0.0, 0, 0)
        print(background_mask[~np.isnan(background_mask)])

        YV_r=np.array(YV).reshape(self.N_crystals_Y, self.N_crystals_X, self.crystal_size_XY_pxl, self.crystal_size_XY_pxl)
        XV_r=np.array(XV).reshape(self.N_crystals_Y, self.N_crystals_X, self.crystal_size_XY_pxl, self.crystal_size_XY_pxl)
        img_norm_r=img_norm[YV_r, XV_r]

        # if self.debug==True:
        #     plt.imshow(img_norm+background_mask)
        #     plt.title("img_norm_r")
        #     plt.colorbar()
        #     plt.show()

        data_normed_noise_crystal_only=np.sum(np.sum(img_norm_r, axis=2), axis=2)#np.sum(np.sum(np.sum(img_norm_r, axis=0), axis=1), axis=1)
        print("img_norm_r.shape=%s"%[img_norm_r.shape])
        print("data_normed_noise_crystal_only.shape=%s"%[data_normed_noise_crystal_only.shape])
        print("np.sum(img_norm_r, axis=2).shape=%s"%[np.sum(img_norm_r, axis=2).shape])

        if self.debug==True:
            data_normed_noise_crystal_only_T=np.transpose(img_norm_r, (0, 2, 1, 3)).reshape(self.N_crystals_Y*self.crystal_size_XY_pxl, self.N_crystals_X*self.crystal_size_XY_pxl)
            plt.imshow(data_normed_noise_crystal_only)
            plt.colorbar()
            plt.show()

            data_normed_noise_crystal_only_mean=np.zeros(data_normed_noise_crystal_only_T.shape)
            signal_summed_over_columns=np.zeros(self.N_crystals_X)
            for j in range(self.N_crystals_X):
                total_counts_per_column=0
                for i in range(self.N_crystals_Y):
                    total_counts_per_crystal=np.sum(data_normed_noise_crystal_only_T[(self.crystal_size_XY_pxl)*i:(self.crystal_size_XY_pxl)*(i+1), self.crystal_size_XY_pxl*j:self.crystal_size_XY_pxl*(j+1)])
                    data_normed_noise_crystal_only_mean[(self.crystal_size_XY_pxl)*i:(self.crystal_size_XY_pxl)*(i+1), self.crystal_size_XY_pxl*j:self.crystal_size_XY_pxl*(j+1)]=total_counts_per_crystal
                    total_counts_per_column+=total_counts_per_crystal
                signal_summed_over_columns[j]=total_counts_per_column
            im=plt.imshow(data_normed_noise_crystal_only_mean)
            im_ratio=data_normed_noise_crystal_only_mean.shape[0]/data_normed_noise_crystal_only_mean.shape[1]
            cb=plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04)
            ax=cb.ax
            cb.ax.set_ylabel('Counts', rotation=270, labelpad=20, fontsize=20)
            plt.xlabel('y (pixels)')
            plt.ylabel('x (pixels)')
            plt.tight_layout()
            plt.savefig('gamma_stack_crystal_array_top.png')
            plt.show()
        return data_normed_noise_crystal_only#-data_normed_noise_crystal_only_summed_div[0]


    def get_measured_signal_summed_over_columns(self, img, get_total_counts=False, get_beam_pointing=False):
        """
        Returns the post-normalisation (i.e. background intensity variations normed out) total counts for each
        column of crystals.
        """

        data_normed_noise_crystal_only_summed=np.sum(self.get_measured_signal_rows_and_cols(img, get_total_counts, get_beam_pointing), axis=0)
        data_normed_noise_crystal_only_summed=data_normed_noise_crystal_only_summed*self.corr_factor_mean
        data_normed_noise_crystal_only_summed_div=data_normed_noise_crystal_only_summed#/np.mean(data_normed_noise_crystal_only_summed)#-1.0
        return data_normed_noise_crystal_only_summed_div[0:self.N_crystals_X_cuttoff]

    def calc_Compton_energy_spec(self, Ec, height):#, offset):
        """
        Returns Compton spectrum given a critical energy
        """
        spec= np.exp(-self.Egamma_MeV_interp/Ec)*(self.Egamma_MeV_interp)**-(2.0/3.0)
        return spec*height#+offset

    def calc_Brems_energy_spec(self, Te_MeV, B):
        gff=3.0**0.5/np.pi*np.exp(self.Egamma_MeV_interp/(2.0*Te_MeV))*kn(0, self.Egamma_MeV_interp/(2.0*Te_MeV))#expi(0.5*(E/Te_MeV)**2)
        dP_dE_brehms=B*1.0/(Te_MeV)**0.5*np.exp(-self.Egamma_MeV_interp/Te_MeV)*gff
        # print("gff=%s"%gff)
        # print("B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)=%s"%(B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)))
        return dP_dE_brehms

    def calc_theoretical_Compton_signal_summed_over_columns(self, x, Ec, height):#, offset):#, offset):#theta, QE, filter_transmission, E_axis_keV, no_sync
        """
        Returns response of crystal array to a gamma beam with energy spectrum well-modelled by a Compton
        spectrum.
        """
        Predicted_signal_summed_over_columns=np.zeros(self.N_crystals_X_cuttoff)
        dN_dE=self.calc_Compton_energy_spec(Ec, height)#, offset)
        for i in range(0, self.N_crystals_X_cuttoff):
            Predicted_signal_summed_over_columns[i]=np.trapz(dN_dE*self.CsIEnergy_ProjZ_interp[:, i], self.Egamma_MeV_interp)
        #Predicted_signal_summed_over_columns_out=Predicted_signal_summed_over_columns#/np.min(Predicted_signal_summed_over_columns)-1.0
        #print(Predicted_signal_summed_over_columns_out.shape)
        return Predicted_signal_summed_over_columns#*height+offset#-Predicted_signal_summed_over_columns_out[0])

    def calc_theoretical_Brems_signal_summed_over_columns(self, x, Ec, height, offset):#theta, QE, filter_transmission, E_axis_keV, no_sync
        """
        Returns response of crystal array to a gamma beam with energy spectrum well-modelled by a Brems
        spectrum.
        """
        Predicted_signal_summed_over_columns=np.zeros(self.N_crystals_X)
        dN_dE=self.calc_Brems_energy_spec(Ec, height)
        for i in range(0, self.N_crystals_X):
            Predicted_signal_summed_over_columns[i]=np.trapz(dN_dE*self.CsIEnergy_ProjZ_interp[:, i], self.Egamma_MeV_interp)
        Predicted_signal_summed_over_columns_out=Predicted_signal_summed_over_columns#/np.min(Predicted_signal_summed_over_columns)-1.0
        return Predicted_signal_summed_over_columns_out#-Predicted_signal_summed_over_columns_out[0])

    def least_sqrs(self, function, guess, measured_signal):
        """
        Fits Compton spectrum given measured counts per column of crystals.
        """
        popt, pcov= opt.curve_fit(function, xdata=None, ydata=measured_signal[0:self.N_crystals_X_cuttoff], p0=guess,  bounds=(0.0, 10**10))
        if (len(measured_signal) > len(guess)) and pcov is not None:
            sigma = np.sqrt(np.diag(pcov))
            return popt, sigma
        else:
            return [None, None], [None, None]

    def generate_estimates(self, params, err_array, percent_std, no_walkers, no_dim):
        guess=np.zeros((no_walkers, no_dim))
        guess[:, 0]=np.random.normal(err_array, err_array*percent_std, no_walkers)
        for j in range(0, len(params)):
            guess[:, j+1]=np.random.normal(params[j], params[j]*percent_std, no_walkers)#np.random.normal(Ec_guess[i-1], Ec_guess[i-1]*0.2, no_walkers)
            # guess[:, i*2+2]=np.random.normal(height[i], height[i]*percent_std, no_walkers)#height[i]*percent_std
        return guess

    def log_likelihood_signal(self, theta, Y_measured, Invert=False):
        sigma = theta[0]
        Y_model=self.calc_theoretical_Compton_signal_summed_over_columns(None, theta[1], theta[2])#, theta[3])#theta, QE, filter_transmission, E_int, no_sync
        if Invert==False:
            Y_model=Y_model*theta[3]#, theta[3])#theta, QE, filter_transmission, E_int, no_sync
        elif Invert==False:
            Y_model=Y_model/theta[3]

        log_likelihood=-np.inf
        is_positive=True
        for i in range(0, len(theta)):
            if theta[i]<0 or np.isnan(theta[i])==True or theta[2]>100.0:
                is_positive=False
        if is_positive==True:
            log_likelihood=np.sum(-0.5*(Y_measured-Y_model)**2/sigma**2-0.5*np.log(2*np.pi)-np.log(sigma))
            if np.isnan(log_likelihood)==True:
                log_likelihood=-np.inf
        return log_likelihood

    def plot_bayes_inferred_spec(self, params):
        sigma_fit1=np.mean(params[:, 0])
        print('sigma_fit1=%s'%sigma_fit1)
        arr_out=get_mean_std_Ec_height_from_MC(params, self.norm_factor)#, ofset_mean, offset_std
        Ec_mean, Ec_std, height_mean, height_std=arr_out[0], arr_out[1], arr_out[2], arr_out[3]
        dN_dEdOmega_fit=np.zeros((len(params), len(self.Egamma_MeV_interp)))
        for j in range(0, len(params)):
        	dN_dEdOmega_fit[j, :]=self.calc_Compton_energy_spec(params[j, 1], params[j, 2])#, params[j, 3])


        print(('Ec_mean=%s +/-%s')%(Ec_mean, Ec_std))
        print(('height_mean=%s+/-%s')%(height_mean, height_std))
        # print(('offset_mean=%s+/-%s')%(offset_mean, offset_std))
        std=2*dN_dEdOmega_fit.std(0)
        mu=dN_dEdOmega_fit.mean(0)
        plt.plot(self.Egamma_MeV_interp, mu, color='b', label='$E_{\gamma c}=%s\pm%s$, $A=%s\pm%s$'%(round(Ec_mean, 2), round(Ec_std, 2), round(height_mean, 3), round(height_std, 3)))
        plt.fill_between(self.Egamma_MeV_interp, mu - std, mu + std, color='b', alpha=0.4)
        plt.xlabel('Energy /MeV')
        plt.ylabel('Fitted Bremsstralung Spectrum')
        plt.legend(loc=0)
        plt.tight_layout()
        return arr_out

    def plot_transmission_inferred(self, params, Y_measured_mean, color_arr=['r', 'b'], diag=None, Invert=False):#, Y_measured_std):

        arr_out=get_mean_std_Ec_height_from_MC(params, self.norm_factor)#, ofset_mean, offset_std
        Ec_mean, Ec_std, height_mean, height_std=arr_out[0], arr_out[1], arr_out[2], arr_out[3]
        caption=['predicted signal %s'%diag, 'measured signal %s'%diag]

        Y_inferred=np.zeros((len(params), self.N_crystals_X_cuttoff))
        for j in range(0, len(params)):
            Y_inferred[j, :]=self.calc_theoretical_Compton_signal_summed_over_columns(None, params[j, 1], params[j, 2])#, params[j, 3])
            if Invert==False:
                Y_inferred[j, :]=Y_inferred[j, :]*params[j, 3]#, theta[3])#theta, QE, filter_transmission, E_int, no_sync
            elif Invert==False:
                Y_inferred[j, :]=Y_inferred[j, :]/params[j, 3]
        Y_inferred_mean=np.mean(Y_inferred, axis=0)#.flatten()
        Y_inferred_std=np.std(Y_inferred, axis=0)#.flatten()

        filter_nos=np.linspace(1, self.N_crystals_X_cuttoff, self.N_crystals_X_cuttoff)

        plt.plot(filter_nos, Y_inferred_mean, color=color_arr[0], label=caption[0])
        plt.fill_between(filter_nos, Y_inferred_mean-Y_inferred_std, Y_inferred_mean+Y_inferred_std, color=color_arr[0], alpha=0.5)
        plt.scatter(filter_nos, Y_measured_mean, color=color_arr[1], marker='.', label=caption[1])
        #plt.vlines(filter_nos, Y_measured_mean-Y_measured_std, Y_measured_mean+Y_measured_std, color='b', alpha=0.5)
        plt.xlabel('CsI Crystal')
        plt.ylabel('Counts')
        plt.legend(loc=0)
        plt.tight_layout()
        return

    def bayesian_opt_spec(self, measured_signal_summed_over_columns):
        model = GaussianProcessRegressorIntegratedY(self.CsIEnergy_ProjZ_interp)
        #inv_CsIEnergy_ProjZ_interp=pinv(self.CsIEnergy_ProjZ_interp)
        #dN_dE_test=np.matmul(measured_signal_summed_over_columns, inv_CsIEnergy_ProjZ_interp)
        # fit the model
        X=self.Egamma_MeV_interp.reshape(-1, 1)
        y=measured_signal_summed_over_columns[0].reshape(-1, 1)
        model.fit(X, y)
        # print(self.Egamma_MeV_interp.shape)
        # for i in range(1, len(measured_signal_summed_over_columns)-1):
        #     print(i)
        #     # x=self.Egamma_MeV_interp.reshape(-1, 1)
        #     # actual=dN_dE_test[i].reshape(-1, 1)
        #     x = opt_acquisition(X, y, model)
        #     # 	# sample the point
        #     actual = objective_fn(x, self.Egamma_MeV_interp, dN_dE_test[i])
        #     est, _ = surrogate(model, [[x]])
        #     X = np.vstack((X, [[x]]))
        #     y = np.vstack((y, [[actual]]))
        #     model.fit(X, y)
        # y_final, std_final=surrogate(model, self.Egamma_MeV_interp.reshape(-1, 1))
        y_final=surrogate(model, self.Egamma_MeV_interp.reshape(-1, 1))

        # for i in range(100):
        # 	# select the next point to sample
        # 	x = opt_acquisition(X, y, model)
        # 	# sample the point
        # 	actual = objective_fn(x, noise_scale)
        # 	# summarize the finding
        # 	est, _ = surrogate(model, [[x]])
        # 	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # 	# add the data to the dataset
        # 	X = np.vstack((X, [[x]]))
        # 	y = np.vstack((y, [[actual]]))
        # 	# update the model
        # 	model.fit(X, y)

        return y_final
