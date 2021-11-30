# based on Cary's interferometry analysis script

import sys, os, pickle
sys.path.append('../../') # this should point to the top level directory
from setup import *
from config import *
from lib.general_tools import *

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from skimage.io import imread
import abel
from scipy.signal import hilbert
from skimage.restoration import unwrap_phase
from skimage.transform import rotate
from scipy.optimize import curve_fit

class interferometry:

    def __init__(self,run_name=None,shot_num=1, diag='LMI', cal_data_path=HOME + '/RR2021/pkg_data'):
        # stores shot information
        self.run_name = run_name
        self.shot_num = shot_num
        self.diag = diag
        self.cal_data_path = cal_data_path

        # make empties
        self.setup_proc()

        # fill with ana_settings data if there
        if run_name is not None:
            if cal_data_path is not None:
                self.cal_file_grab()

    def setup_proc(self):
        # set up placeholder defaults
        nrows, ncols = 492, 656 # standard Manta chip
        self.raw_img = np.full((nrows, ncols), np.nan)
        self.fringes_img = np.full((nrows, ncols), np.nan)
        self.phase = np.full((nrows, ncols), np.nan)
        self.int_ne_dl = np.full((nrows, ncols), np.nan)
        self.ne = np.full((nrows, ncols), np.nan)

        # default options
        self.ana_settings_path = None
        self.channel_roi = [0, nrows, 0, ncols] #subsection referenced from fringes_roi
        self.fringes_roi = [0, nrows, 0, ncols]
        self.fmask_params = [113,30,16,20,8,8]  # mask for fft retrieval
        self.fourier_mask = np.ones((nrows, ncols))
        self.channel_mask = np.ones_like(self.phase)
        self.centre = (0., 0.)
        self.ref_shots = []      # add in shot references - default empty list

        # constants for integration
        self.lambda_0 = 800.0 * 1e-9
        self.m_e = constants.m_e
        self.e = constants.e

    def get_filepath(self, l):
        """l is list/tuple of date (str), run (str), shot number (int)

        if fails, it returns None
        """
        path = None
        date,run,shot_num = l
        run_name = date + '/' + run
        shot_num = str(shot_num)
        ss = 'Shot' + '0'*(3 - len(shot_num)) + shot_num
        file_ext = '.TIFF'
        diag = self.diag

        # grab data from both diagnostics - dataholder
        path = ROOT_DATA_FOLDER + '/' + diag + '/' + date + '/' + run + '/' + ss + file_ext

        if os.path.isfile(path) == False:
            # fix for some file mislabelling
            burstdir = 'Burst' + '0'*(3 - len(str(shot_num))) + str(shot_num)
            path = ROOT_DATA_FOLDER + '/' + diag + '/' + date + '/' + run + '/' + burstdir + '/' + 'Shot001' + file_ext

        return path

    def get_raw_image(self, path, overwrite=True):
        """pure raw image, backup
        """
        try:
            raw_img = np.array(imread(path), dtype=float)
            if overwrite:
                self.raw_img = np.copy(raw_img)
        except FileNotFoundError as e:
            print(e)
        return np.copy(raw_img)

    def get_fringe_image(self, raw_img, overwrite=True):
        """raw image cropped to just where fringes are
        """
        top, bottom, left, right = self.fringes_roi
        Z = np.copy(raw_img)
        Z = Z[top:bottom, left:right]
        if overwrite:
            self.fringes_img = np.copy(Z)
        return Z

    def rotate_image(self, image, theta=0.0):
        """theta in deg
        """
        if np.isnan(theta):
            theta = 0.0
        im = np.copy(image)
        im = rotate(im, theta)
        return im

    def FFT(self, image, shift=True):
        """returns f_x axis, f_y axis and fourier transform. All 2D
        """
        Z = np.copy(image)
        Z = Z - np.mean(Z)
        F_Z = np.fft.fft2(Z)
        if shift==True:
            F_Z = np.fft.fftshift(F_Z)

        nrows, ncols = F_Z.shape
        u,v = np.arange(ncols), np.arange(nrows)
        U,V = np.meshgrid(u,v)
        return U, V, F_Z

    def iFFT(self, f_image, shift=True):
        """need to invert shift first before ifft
        """
        Z = np.copy(f_image)
        if shift==True:
            Z = np.fft.ifftshift(Z)
        Z = np.fft.ifft2(Z)

        nrows, ncols = Z.shape
        x,y = np.arange(ncols), np.arange(nrows)
        X,Y = np.meshgrid(x,y)
        return X, Y, Z

    def Gauss(self, X,Y, masked_params, overwrite = True):
        """Normalised 2D super-gaussian for masking fourier space
        mask_params = [ux, uy, ox, oy, nx, ny]

        """
        ux, uy, ox, oy, nx, ny = masked_params
        index_x = ((X-ux)**nx/(2.*ox**nx))
        index_y = ((Y-uy)**ny/(2.*oy**ny))
        f = np.exp( -(index_x + index_y))
        area = np.trapz(np.trapz(f, X[0]), Y[:,0])

        if overwrite == True:
            self.fourier_mask = f/area

        return f/area

    def hilbert_retrieval(self, image, w = 16):
        """ hilbert transform (line by line) phase retrieval """

        nrows, ncols = image.shape

        phase_wrapped = np.zeros(image.shape)

        for row in range(nrows):

            lineout = image[row]

            # centre signal at 0
            moving_avg = np.convolve(lineout, np.ones(w), 'same') / w
            lineout_centred = lineout - moving_avg

            # find phase
            analytic_signal = hilbert(lineout_centred)

            phase_wrapped[row] = np.angle(analytic_signal)

        phase = unwrap_phase(phase_wrapped)

        return phase

    def calc_phase(self, fringes_img, method, w=16, shift_in=True, shift_out=True, ref_Z=None, overwrite=True):
        """ run entire phase retrieval analysis """

        if ref_Z is None:
            ref_Z = np.ones_like(fringes_img)

        if method == 'hilbert':
            Z = self.hilbert_retrieval(fringes_img, w)

            phase_shift = Z - ref_Z

        elif method == 'fft' or 'FFT':
            Z = np.copy(fringes_img)
            _, _, F_Z = self.FFT(Z, shift_in)

            F_Z_masked = F_Z * self.fourier_mask

            _, _, Z = self.iFFT(F_Z_masked, shift_out)

            phase_wrapped = np.angle(Z/ref_Z)
            phase_shift = unwrap_phase(phase_wrapped)

        if overwrite:
            self.phase = np.copy(phase_shift)

        return phase_shift

    def get_ref_phase(self, ref_shots, method, shift_in=True, shift_out=True):
        """Given list of them, return average phase map
        """

        ref_filepaths = [self.get_filepath(i) for i in ref_shots]
        ref_raw_imgs = [self.get_raw_image(p, overwrite=False) for p in ref_filepaths]
        ref_fringe_imgs = [self.get_fringe_image(ri, overwrite=False) for ri in ref_raw_imgs]

        if method == 'fft' or 'FFT':
            ref_F_Zs = [self.FFT(rfi, shift_in)[-1] for rfi in ref_fringe_imgs]
            ref_F_Z_maskeds = [rfz * self.fourier_mask for rfz in ref_F_Zs]
            ref_Zs = [self.iFFT(rfz, shift_out)[-1] for rfz in ref_F_Z_maskeds]

        elif method == 'hilbert':
            ref_Zs = [self.hilbert_retrieval(rfi) for rfi in ref_fringe_imgs]

        ref_Z = np.mean(ref_Zs, axis=0)

        return ref_Z

    def calc_channel_angle(self, phase=None):
        """ Find the angle of the plasma channel from the phase map """
        if phase is None:
            phase = np.copy(self.phase)

        # neglect 10 px from channel edges
        col_min, col_max = self.channel_roi[2]+10, self.channel_roi[3]-10

        # At every x', find y' of max phase shift
        y_prime_max = phase.argmax(axis=0)[col_min:col_max]
        phi_max = phase.max(axis=0)[col_min: col_max]
        x_prime = np.arange(col_min, col_max)  #np.arange(phase.shape[1])

        # fit a line through that
        coeff = np.polyfit(x_prime, y_prime_max, 1)
        poly = np.poly1d(coeff)

        m,c = coeff
        theta = np.degrees(np.arctan(m))

        print('theta=', theta)
        #plt.figure()
        #plt.plot(x_prime,y_prime_max)

        return theta

    def phase_to_SI(self, phase = None, overwrite=True):
        """converts captured phase to integrated n_e (cm-3) along line of sight
        """
        if phase is None:
            phase = np.copy(self.phase)
        mperpixel = self.umperpixel * 1e-6
        int_ne_dl_SI = phase * self.m_e * 1e7 / (self.e**2 * self.lambda_0 ) / mperpixel
        int_ne_dl = int_ne_dl_SI * (1e-6) # per cm3

        if overwrite:
            self.int_ne_dl = np.copy(int_ne_dl)

        return int_ne_dl

    def calc_channel_mask(self, ROI=None, overwrite=True):
        """ plasma channel region of interest """
        if ROI is None:
            ROI = self.channel_roi
        phase = self.phase

        mask = np.zeros_like(phase)
        mask[ROI[0]:ROI[1], ROI[2]:ROI[3]] = 1.0

        if overwrite:
            self.channel_mask = np.copy(mask)
        return mask

    def phase_bg_cor(self, phase, overwrite=True):
        ROI = self.channel_roi

        bg_phasemap = np.copy(phase)
        bg_phasemap[ROI[0]:ROI[1], ROI[2]:ROI[3]] = np.nan  # ignore plasma channel

        bg_phase = np.nanmean(bg_phasemap) # mean background phase
        print('bg phase:', bg_phase)

        # subtract background phase
        phase_cor = phase - bg_phase

        if overwrite == True:
            self.phase = np.copy(phase_cor)

        return phase_cor

    def correct_phase_polarity(self, phase=None, overwrite=True):
        if phase is None:
            phase = self.phase
        c_mask = np.where(self.channel_mask==1.0, 1.0, np.nan)
        not_c_mask = np.where(c_mask==1.0, np.nan, 1.0)

        inner = np.nanmean(phase * c_mask)
        outer = np.nanmean(phase * not_c_mask)

        if inner < outer:
            phase = -1.0 * phase

        if overwrite:
            self.phase = np.copy(phase)

        return phase

    def calc_phase_centre(self, phase=None, method=None, axes=(0,1), overwrite=True):
        """finds phase central axis for abel inversion based on just the channel.
        centre will be for an image with the channel vertical (what PyAbel wants)

        method kwarg passed onto PyAbel's find_origin function.
        If given as a list, then it will do this method to each axis.
        axes is 0, 1 or (0,1) for centering. 0 is horizontal, 1 is vertical
        """
        if method is None:
            method = 'convolution'

        if phase is None:
            phase = np.copy(self.phase)
        c_mask = np.copy(self.channel_mask)
        phase_roi = phase * c_mask

        phase = phase.T #abel wants the inversion axis to be vertical
        phase_roi = phase_roi.T

        if type(method)==list:
            centre_x, _ = abel.tools.center.find_origin(phase, method=method[0], axes=0)
            _, centre_y = abel.tools.center.find_origin(phase, method=method[1], axes=1)
            centre = (centre_x, centre_y)

        else:
            centre = abel.tools.center.find_origin(phase_roi, method=method, axes=axes)

        if overwrite:
            self.centre = centre

        return centre


    def abel_invert_int_ne_dl(self, int_ne_dl = None, method=None, overwrite=True):
        """method kwarg passed on to PyAbel's Transform function
        """
        if method is None:
            method = 'direct'
        if int_ne_dl is None:
            int_ne_dl = np.copy(self.int_ne_dl)

        int_ne_dl = int_ne_dl.T # PyAbel wants the inversion axis vertical
        centre = np.copy(self.centre)

        centred_image = abel.tools.center.set_center(int_ne_dl, origin=centre, axes=1)

        ne = abel.Transform(centred_image, method=method,
                                             direction='inverse').transform

        ne = ne.T

        if overwrite:
            self.ne = np.copy(ne)

        return ne

    def calc_phase_noise(self):
        ''' quantifies the background noise of the phase map '''

        # define channel and background
        c_mask = np.where(self.channel_mask==1.0, 1.0, np.nan)
        not_c_mask = np.where(c_mask==1.0, np.nan, 1.0)

        # histogram of background pixels to characterise noise
        background = self.phase * not_c_mask
        background = background[~np.isnan(background)].flatten()
        distribution, bins = np.histogram(background, bins=50)
        bin_l = np.delete(bins,-1)
        bin_r = np.delete(bins,0)
        bin_c = (bin_l + bin_r)/2

        # fit gaussian and take the standard deviation as the uncertainty
        popt_b, pcov_b = curve_fit(gaussian, bin_c, distribution)
        phase_noise = np.abs(popt_b[0])

        return phase_noise

    def calc_error(self):
        ''' quantifies uncertainty on the density as the noise of the abel inverted result '''

        # define channel and background
        # abel inversion centers the channel vertically
        channel_mid = int(self.ne.shape[0]/2)
        channel_width = int((self.channel_roi[1] - self.channel_roi[0])/2)
        ne_channel_roi = [channel_mid - channel_width, channel_mid + channel_width, self.channel_roi[2],
                         self.channel_roi[3]]
        ne_channel = self.calc_channel_mask(ne_channel_roi, overwrite=False)
        bg_mask = np.where(ne_channel==1.0, np.nan, 1.0)
        # disregard the 0s added when centering the channel
        bg_mask[self.ne == 0] = np.nan

        # histogram of background pixels to characterise noise
        background = self.ne * bg_mask
        background = background[~np.isnan(background)].flatten()
        distribution, bins = np.histogram(background, bins=50)
        bin_l = np.delete(bins,-1)
        bin_r = np.delete(bins,0)
        bin_c = (bin_l + bin_r)/2

        # fit gaussian and take the standard deviation as the uncertainty
        popt_b, pcov_b = curve_fit(gaussian, bin_c, distribution, p0=[2e17,1800])
        noise = np.abs(popt_b[0])

        return noise

    def get_ne(self, raw_data=None, path=None, retrieval_method='FFT',
    calc_centre_method=['convolution', 'gaussian'],
    abel_method=None, calc_error = False, theta=None):
        """  do all processing in one function """

        if raw_data is None:
            self.get_raw_image(path)
        else:
            self.raw_img = np.copy(raw_data)
        self.get_fringe_image(self.raw_img)

        if retrieval_method == 'FFT' or 'fft':
            nrows, ncols = self.fringes_img.shape
            u,v = np.arange(ncols), np.arange(nrows)
            U,V = np.meshgrid(u,v)
            self.Gauss(U, V, self.fmask_params)

        ref_Z = self.get_ref_phase(self.ref_shots, method=retrieval_method)

        self.calc_phase(self.fringes_img, ref_Z = ref_Z, method=retrieval_method)

        # post processing for abel
        self.calc_channel_mask()
        self.correct_phase_polarity()
        self.phase_bg_cor(self.phase)

        if theta is None:
            theta = self.calc_channel_angle()
            print('theta:', theta)
        self.phase = self.rotate_image(self.phase, theta)

        self.calc_phase_noise()

        # abel inversion
        centre = self.calc_phase_centre(method=calc_centre_method)
        self.phase_to_SI()
        self.abel_invert_int_ne_dl(method=abel_method)

        if calc_error:
            n_uncertainty = self.calc_error()
            #phase_uncertainty = self.calc_phase_noise()
            #n_uncertainty = phase_uncertainty * np.max(self.ne) / np.max(self.phase*self.channel_mask) / np.sqrt(2)
            return np.copy(self.ne), n_uncertainty
        else:
            return np.copy(self.ne)


def gaussian(x, w, A):

    return A* np.exp(-((x)/w)**2)
