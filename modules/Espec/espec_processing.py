
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import median_filter
from scipy.io import loadmat
from scipy.interpolate import interp1d, RectBivariateSpline
import sys
sys.path.append('../..')
sys.path.append('../')

from lib.general_tools import load_object

espec1_screen_to_jet=1.74
print(espec1_screen_to_jet)
class Espec_proc():
    """ Object for handling espec_high analysis
        hardcoded fC_per_count value from rough calibration by Matt on 28th August 2020
        args
            tForm_filepath is the path of the image warp cv2 perspective transform
                contains x_mm and y_mm which are the real spatial axes of the un-warped image
            Espec_cal_filepath is the conversion from x_mm to energy in MeV
        kwargs:
            img_bkg = can be an espec_high image or a single value or None to disable this subtraction method
            use_median = boolean, if True then the median value will be subtracted from the raw image (crude background noise subtraction)
            kernel_size = (int,int) or None: used for median 2d filter
    """
    fC_per_count = 0.0012019877034770447
    def __init__(self,tForm_filepath,Espec_cal_filepath,img_bkg=None,use_median=True,kernel_size=None ):
        # warping info
        tForm = load_object(tForm_filepath)
        if 'fC_per_count' in tForm.keys():
            self.fC_per_count = tForm['fC_per_count']
            print('self.fC_per_count=%s'%self.fC_per_count)
        self.imgArea0 = np.abs(tForm['imgArea0'])
        self.H = tForm['H']
        self.newImgSize = tForm['newImgSize']
        self.imgArea1 = np.abs(tForm['imgArea1'])
        self.screen_x_mm = tForm['x_mm']
        self.screen_y_mm = tForm['y_mm']
        self.screen_dx = np.mean(np.diff(self.screen_x_mm))
        self.screen_dy = np.mean(np.diff(self.screen_y_mm))
        self.screen_dist_m=espec1_screen_to_jet

        # dispersion calibration
        Espec_cal = loadmat(Espec_cal_filepath)

        self.spec_x_mm = Espec_cal['spec_x_mm'].flatten()
        self.spec_MeV = Espec_cal['spec_MeV'].flatten()
        iSel = np.isfinite(self.spec_MeV)
        self.spec_cal_func = interp1d(self.spec_x_mm[iSel], self.spec_MeV[iSel],
            kind='linear', copy=True, bounds_error=False, fill_value=0)

        self.screen_energy = self.spec_cal_func(self.screen_x_mm)
        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                g = -np.gradient(self.screen_x_mm,self.screen_energy)
                g[np.isfinite(g)<1]=0
        self.dispersion = g
        # background subtraction options
        self.img_bkg=img_bkg
        self.use_median=use_median
        self.kernel_size=kernel_size

        # energy axis for final spectra
        self.eAxis_MeV = np.linspace(250,2500,num=4000)
        self.dE_MeV = np.mean(np.diff(self.eAxis_MeV))


    def espec_warp(self,img_raw):
        """ calc transformed image using tForm file and cv2 perspective transform
        """
        img = self.espec_background_sub(img_raw)

        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                imgCountsPerArea = img/self.imgArea0
        imgCountsPerArea[self.imgArea0==0] =0
        imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
        imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

        im_out = cv2.warpPerspective(imgCountsPerArea, self.H, self.newImgSize)*self.imgArea1
        return im_out

    def espec_file2screen(self,file_path):
        """ Takes a data file and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """
        img_raw = plt.imread(file_path)
        img_pC_permm2 = self.espec_data2screen(img_raw)
        return img_pC_permm2

    def espec_data2screen(self,img_raw):
        """ Takes raw data (previous opened from data file) and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """
        img_warp= self.espec_warp(img_raw.astype(float))
        img_pC_permm2 = img_warp*self.fC_per_count/self.imgArea1 *1e-3
        return img_pC_permm2

    def espec_data2screen_counts_per_mm(self,img_raw):
        """ Takes raw data (previous opened from data file) and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """

        img_warp= self.espec_warp(img_raw)
        return img_warp/self.imgArea1

    def espec_background_sub(self,img_raw):
        """ background subtraction method
        """
        if self.img_bkg is None:
            img_sub = img_raw
        else:
            img_sub = img_raw-self.img_bkg

        if self.use_median:
            img_sub = img_sub -np.median(img_sub)

        if self.kernel_size is not None:
            img_med = median_filter(img_sub,self.kernel_size)
            threshold=5.0*np.std(img_sub)
            diff=abs(img_sub-img_med)
            img_sub[diff>threshold] = img_med[diff>threshold]

        return img_sub

    def espec_screen2spec(self,img_screen):
        """ convert image to spectrum
        Uses 1d interpolation along horrizontal axis (1)
        """
        spec = img_screen*self.dispersion
        spec_func = interp1d(self.screen_energy,spec, bounds_error=False, fill_value=0)
        spec_pC_per_mm_per_MeV = spec_func(self.eAxis_MeV)
        return spec_pC_per_mm_per_MeV

    def total_charge(self,img_raw):
        """ Integrates the screen image to get the total charge
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        return np.sum(img_pC_permm2)*self.screen_dx*self.screen_dy

    def energy_spectrum(self,img_raw):
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        spec_J_per_mm_per_MeV = self.eAxis_MeV*spec_pC_per_mm_per_MeV
        return spec_J_per_mm_per_MeV

    def total_beam_energy(self,img_raw):
        """ Integrates the spectrum to get total beam energy
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        W_b = np.sum(np.sum(spec_pC_per_mm_per_MeV,axis=0)*self.screen_dy*self.dE_MeV *self.eAxis_MeV*1e-12*1e6)
        return W_b# in J

    def total_beam_energy_squared(self,img_raw):
        """ Integrates the spectrum and multiplies the charge content in each bin by the energy squared (expected radiation scaling)
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        QE_squared = np.sum(np.sum(spec_pC_per_mm_per_MeV,axis=0)*self.screen_dy*self.dE_MeV *self.eAxis_MeV*self.eAxis_MeV*1e-12*1e6*1e6)
        return QE_squared

    def spec_y_summed(self,img_raw):
        """ Integrates the spectrum and multiplies the charge content in each bin by the energy squared (expected radiation scaling)
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        spec_pC = np.sum(spec_pC_per_mm_per_MeV,axis=0)*self.screen_dy*self.dE_MeV*1e-12
        return spec_pC/np.trapz(spec_pC, self.eAxis_MeV)

    def mean_and_std_beam_energy(self,img_raw):
        """ Gets mean and std of electron energy. Returns electron energy at 90th percentile of charge distribution.
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        img_pC_per_MeV = np.trapz(self.espec_screen2spec(img_pC_permm2), self.screen_y_mm, axis=0)
        spec_charge_dist= img_pC_per_MeV/np.trapz(img_pC_per_MeV, self.eAxis_MeV)
        spec_charge_dist[spec_charge_dist<=0]=0.0
        mean_energy = np.trapz(spec_charge_dist*self.eAxis_MeV, self.eAxis_MeV)
        #Exp_energy_sqrd = np.trapz(spec_charge_dist*self.eAxis_MeV**2, self.eAxis_MeV)
        variance = np.trapz(spec_charge_dist*(self.eAxis_MeV-mean_energy)**2, self.eAxis_MeV)

        div=10.0
        N=int(len(spec_charge_dist)/div)
        percentile, energy=np.zeros(N), np.zeros(N)
        target_percentile=0.9

        for i in range(0, N):
            percentile[i]=np.trapz(spec_charge_dist[0:len(self.eAxis_MeV)-1-int(div)*i], self.eAxis_MeV[0:len(self.eAxis_MeV)-1-int(div)*i])
            energy[i]=self.eAxis_MeV[len(self.eAxis_MeV)-1-int(div)*i]
            if percentile[i]<target_percentile-0.05:
                break
        percentile_cut=percentile[percentile!=0.0]
        energy_cut=energy[percentile!=0.0]
        energy_at_90th_percentile=np.interp(target_percentile, percentile_cut, energy_cut)
        return np.array([mean_energy, variance**0.5, energy_at_90th_percentile])

    def beam_divergence_y(self,img_raw):
        """ Returns beam divergence in y (non-dispersed axis)
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        div_y_mrad=self.screen_y_mm/self.screen_dist_m
        img_pC_permm=np.trapz(img_pC_permm2, self.screen_x_mm, axis=1)
        img_charge_dist = img_pC_permm/np.trapz(img_pC_permm, div_y_mrad)
        img_charge_dist[img_charge_dist<0.009]=0.0
        mean_div_y_mrad = np.trapz(img_charge_dist*div_y_mrad, div_y_mrad)
        var_div_y_mrad_sqrd= np.trapz(img_charge_dist*div_y_mrad**2, div_y_mrad)-mean_div_y_mrad**2
        return var_div_y_mrad_sqrd**0.5

    def off_axis_tilt(self,img_raw):
        """ Returns beam divergence in y (non-dispersed axis)
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        Ny,Nx = np.shape(img_pC_permm2)

        y = self.screen_y_mm/self.screen_dist_m#np.arange(Ny)
        x = self.screen_x_mm/self.screen_dist_m#np.arange(Nx)
        img_centre_y=np.mean(y)
        img_charge_disty=img_pC_permm2/np.trapz(img_pC_permm2, y, axis=0)
        img_pC_permm_y=np.trapz(img_charge_disty*y.reshape(-1,1), y, axis=0)
        xy_fit_coeffs=np.polyfit(x, img_pC_permm_y, 1)
        yfit=xy_fit_coeffs[1]+xy_fit_coeffs[0]*x
        angle_to_horz=np.arctan((np.amax(yfit)-np.amin(yfit))/(np.amax(x)-np.amin(x)))

        mean_pos_y=np.trapz(np.trapz(img_pC_permm2, x, axis=1)*y, y)/np.trapz(np.trapz(img_pC_permm2, y, axis=0), x)
        return x, yfit, angle_to_horz*1000.0, (mean_pos_y-img_centre_y)

ESpec_high_proc=Espec_proc

def load_espec_image(file_path):
    return imread(file_path).astype(float)


class Espec_ang_proc():
    """ Object for handling espec_high analysis
        hardcoded fC_per_count value from rough calibration by Matt on 28th August 2020
        args
            tForm_filepath is the path of the image warp cv2 perspective transform
                contains x_mm and y_mm which are the real spatial axes of the un-warped image
            Espec_cal_filepath is the conversion from x_mm to energy in MeV
        kwargs:
            img_bkg = can be an espec_high image or a single value or None to disable this subtraction method
            use_median = boolean, if True then the median value will be subtracted from the raw image (crude background noise subtraction)
            kernel_size = (int,int) or None: used for median 2d filter
    """


    def __init__(self,tForm_filepath,Espec_cal_filepath,img_bkg=None,use_median=True,kernel_size=None ):
        # warping info
        tForm = load_object(tForm_filepath)
        self.tForm = tForm
        self.imgArea0 = np.abs(tForm['imgArea0'])
        self.H = tForm['H']
        self.newImgSize = tForm['newImgSize']
        self.imgArea1 = np.abs(tForm['imgArea1'])
        self.screen_x_mm = tForm['x_mm']
        # self.screen_x_mm = tForm['deflection_mm'] # not used in this experiment
        self.screen_y_mm = tForm['y_mm']
        if 'fC_per_count' in tForm.keys():
            self.fC_per_count = tForm['fC_per_count']
        else:
            self.fC_per_count = 1

        # the direction of the arrays wants to be +ve for RectBivariateSpline
        self.screen_dx = np.mean(np.diff(self.screen_x_mm))
        if self.screen_dx<0:
            self.rev_x = True
            self.screen_dx = -self.screen_dx
            self.screen_x_mm = self.screen_x_mm[::-1]
        else:
            self.rev_x = False
        self.screen_dy = np.mean(np.diff(self.screen_y_mm))
        if self.screen_dy<0:
            self.rev_y = True
            self.screen_dy = -self.screen_dy
            self.screen_y_mm = self.screen_y_mm[::-1]
        else:
            self.rev_y = False

        # dispersion calibration
        self.Espec_cal = load_object(Espec_cal_filepath)
        # create functions to give energy as function of position and angle
        x,y =self.Espec_cal['spec_x_mm'],self.Espec_cal['t_mrad']
        self.E_MeV_theta_mm = RectBivariateSpline(y,x,self.Espec_cal['E_x_t'])
        self.x_min_func = interp1d(self.Espec_cal['t_mrad'],self.Espec_cal['x_lim'][:,0])
        self.x_max_func = interp1d(self.Espec_cal['t_mrad'],self.Espec_cal['x_lim'][:,1])

        # create function to give x-position as function of energy and angle
        x,y = self.Espec_cal['X_e_t_E_MeV'],self.Espec_cal['t_mrad']
        self.x_mm_theta_MeV = RectBivariateSpline(y,x,1e3*self.Espec_cal['X_e_t'],kx=1, ky=1)
        self.E0 = 1000

        # background subtraction options
        self.img_bkg=img_bkg
        self.use_median=use_median
        self.kernel_size=kernel_size

        # energy axis for final spectra
        self.eAxis_MeV = np.linspace(300,2300,num=1000)
        self.dE_MeV = np.mean(np.diff(self.eAxis_MeV))


    def espec_warp(self,img_raw):
        """ calc transformed image using tForm file and cv2 perspective transform
        """
        img = self.espec_background_sub(img_raw)

        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                imgCountsPerArea = img/self.imgArea0
        imgCountsPerArea[self.imgArea0==0] =0
        imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
        imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

        im_out = cv2.warpPerspective(imgCountsPerArea, self.H, self.newImgSize)*self.imgArea1
        if self.rev_x:
            im_out = np.fliplr(im_out)
        if self.rev_y:
            im_out = np.flipud(im_out)
        return im_out

    def espec_file2screen(self,file_path):
        """ Takes a data file and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """
        img_raw = load_espec_image(file_path)
        img_pC_permm2 = self.espec_data2screen(img_raw)
        return img_pC_permm2

    def espec_data2screen(self,img_raw):
        """ Takes raw data (previous opened from data file) and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """

        img_warp= self.espec_warp(img_raw)
        img_pC_permm2 = img_warp*self.fC_per_count/self.imgArea1 *1e-3
        return img_pC_permm2

    def espec_background_sub(self,img_raw):
        """ background subtraction method
        """
        if self.img_bkg is None:
            img_sub = img_raw*1.0
        else:
            img_sub = img_raw-self.img_bkg

        if self.use_median:
            img_sub = img_sub -np.median(img_sub)

        if self.kernel_size is not None:
            img_sub = median_filter(img_sub,self.kernel_size)

        return img_sub

    def espec_screen2spec(self,img_screen,theta):
        """ convert image to spectrum
        Uses 1d interpolation along horrizontal axis (1)
        """
        if np.size(theta)==1:
            theta = theta*np.ones_like(self.screen_x_mm)

        x0_ind = np.nanargmin((self.eAxis_MeV-self.E0 )**2)

        # interpolate X_e_t to find E(x)
        x = self.x_mm_theta_MeV.ev(theta,self.eAxis_MeV)
        iSel = np.isfinite(x)
        x[iSel<1] = -1

        # select by gradient
        x_grad = np.gradient(x)
        grad_sign = np.sign(x_grad[x0_ind])
        iSel = iSel*((grad_sign*x_grad)>0)*(x>0)



        if np.max(iSel[:x0_ind]<1)>0:
            i_min = np.max(np.argwhere(iSel[:x0_ind]<1))+1
        else:
            i_min = np.min(np.argwhere(iSel>0))


        if np.min(iSel[i_min:])==0:
            i_max = np.min(np.argwhere((iSel[i_min:]<1)))-1 + i_min

        else:
            i_max = len(iSel)-1

        iSel = np.arange(len(iSel))
        iSel = (iSel>=i_min)*(iSel<=i_max)
        x_lims = sorted([x[i_max],x[i_min]])

        # # interpolate to screen_x

        screen_energy = interp1d(x[iSel],self.eAxis_MeV[iSel],bounds_error=False,
                                fill_value=0)(self.screen_x_mm)

        # iSel2 = (Espec1_proc.screen_x_mm>x_min)*(Espec1_proc.screen_x_mm<x_max)

        # # x_max = self.x_max_func(theta)
        # # x_min = self.x_min_func(theta)


        # # screen_energy = self.E_MeV_theta_mm.ev(theta,self.screen_x_mm).flatten()
        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                dispersion = grad_sign*np.gradient(self.screen_x_mm,screen_energy)

                dispersion[np.isfinite(dispersion)<1]=0

        spec_good = (self.screen_x_mm>x_lims[0])*(self.screen_x_mm<x_lims[1])
        # ind = np.arange(len(screen_energy))
        # spec_good = spec_good *(ind>np.max(np.where((dispersion<0))))

        spec = img_screen*dispersion*spec_good
        spec_func = interp1d(screen_energy,spec, bounds_error=False, fill_value=0)
        spec_pC_per_mm_per_MeV = spec_func(self.eAxis_MeV)
        self.screen_energy = screen_energy
        return spec_pC_per_mm_per_MeV
