from config import *
import numpy as np
import matplotlib.pyplot as plt

from lib.general_tools import *
from skimage.io import imread
import abel

from skimage.restoration import unwrap_phase
from skimage.transform import rotate
from scipy import constants

from scipy.optimize import curve_fit

class Interferometry:
    def __init__(self,run_name=None,shot_num=1,cal_data_path=HOME+'/calib/Probe',diag='LMI'):
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
        
        self.reset()
        
        # some funcitonality varies between different version of abel
        self.abel_version = str(abel.__version__)
        
    def setup_proc(self):
        # default options
        nrows, ncols = 492, 656 # standard Manta chip
        
        # default options overwritten by ana_settings file
        self.ana_settings_path = None
        self.ana_settings_loaded = {}
        self.channel_roi = [0, nrows, 0, ncols] #subsection referenced from fringes_roi
        self.fringes_roi = [0, nrows, 0, ncols]
        self.fourier_mask = np.ones((nrows, ncols))
        self.phase = np.zeros((nrows, ncols))
        self.channel_mask = np.ones_like(self.phase)
        self.raw_img_extent = [0, ncols, nrows, 0]  # extent=[xmin, xmax, ymax, ymin])
        self.fringes_img_extent = [0, ncols, nrows, 0]
        self.fringes_img_extent_mm = [0, ncols, nrows, 0]
        self.raw_img = np.full((nrows, ncols), fill_value=np.nan)
        self.raw_img_x_mm = np.arange(ncols)
        self.raw_img_y_mm = np.arange(nrows)
        self.ref_shots = []      # add in shot references - default empty list
        self.ref_raw_imgs = None

        self.channel_width = 5
        self.umperpixel = 1.0
        
        self.centre_method = 'convolution'
        self.inversion_method = 'direct'
        self.shift_in = True
        self.shift_out = True
        self.apply_no_ref_correction = False
        self.n_std_threshold = 0.75
        self.invert_phase = False
        
        self.fixed_channel_centre = False		
		
        # constants for integration
        self.lambda_0 = 800.0 * 1e-9
        self.m_e = constants.m_e
        self.e = constants.e
        
        
    def reset(self):
        """Clean slate between grabbing shot data
        Changed so doesn't reset any vars set by ana_settings
        """
        # set up placeholder defaults 
        nrows, ncols = 492, 656 # standard Manta chip
        self.raw_img = np.full((nrows, ncols), np.nan)
        #self.raw_img_x_mm = np.arange(ncols)
        #self.raw_img_y_mm = np.arange(nrows)
        
        left, right, bottom, top = self.fringes_img_extent
        nrows, ncols = bottom - top, right - left
        self.fringes_img = np.full((nrows, ncols), np.nan)
        self.phase_original = np.full((nrows, ncols), np.nan)
        self.phase = np.full((nrows, ncols), np.nan)
        self.int_ne_dl = np.full((nrows, ncols), np.nan)
        self.ne = np.full((nrows, ncols), np.nan)
        self.centre = (0., 0.)
        self.channel_angle = np.nan
        self.channel_offset = np.nan        

        self.ne_lineout = np.full((ncols), np.nan)  
        #self.ne_x_mm = np.arange(ncols) * 1.0
        #self.ne_y_mm = np.arange(nrows) * 1.0        
    
    def cal_file_grab(self):
        # read in ana_settings
        try:
            #t = choose_cal_file(self.run_name, self.shot_num, self.diag,
            #            self.diag + '_ana_settings', cal_data_path=self.cal_data_path)
            
            t = choose_cal_file(self.run_name, self.shot_num, self.diag,
                        self.diag + '_ana_settings', cal_data_path=self.cal_data_path)
            print('ana_settings file chosen is: ', t)
            self.ana_settings_path = t    
            ana_settings = load_object(t)
            self.ana_settings_loaded = dict(ana_settings)
            
            for key, value in ana_settings.items():
                # overwrite placeholders with values in pickle file
                setattr(self, key, value)

            if self.ref_shots != []:
                self.ref_raw_imgs = self.get_ref_images(self.ref_shots)

            dx_mm = self.umperpixel*1e-3
            self.raw_img_x_mm = np.copy(self.raw_img_x_mm) * dx_mm
            self.raw_img_y_mm = np.copy(self.raw_img_y_mm) * dx_mm

            top, bottom, left, right = self.fringes_roi
            self.fringes_img_extent = [left, right, bottom, top]
            self.fringes_img_extent_mm = np.array([left, right, bottom, top]) * dx_mm            
            #print('self.fringes_roi', self.fringes_roi)
            #print('self.fringes_extent', self.fringes_img_extent)
            
            nrows, ncols = bottom-top, right-left
            self.ne = np.full((nrows, ncols), np.nan)
            self.ne_x_mm = np.arange(left,right) * dx_mm
            self.ne_y_mm = np.arange(top,bottom) * dx_mm
            
            """
            dx_mm = self.umperpixel * 1e-3
            self.fringes_img_extent_mm = np.copy(self.fringes_img_extent) * dx_mm
            l,r,b,t = np.copy(self.fringes_img_extent)
            print('self.fringes_img_extent', self.fringes_img_extent)
            
            self.ne_x_mm = np.arange(l,r) * dx_mm
            self.ne_y_mm = np.arange(b,t) * dx_mm
            """
            
        except (IndexError, ValueError) as e:
            print(e)
            print('Filename error (?) in the folder: %s' % (self.cal_data_path + '/' + self.diag + '/'))
            # break rest of setup

    def get_filepath(self, l, diag=None):
        """l is list/tuple of date (str), run (str), shot number (int)
        
        if fails, it returns None
        """
        path = None

        if diag is None:
            diag = self.diag
        
        date,run,shot_num = l
        run_name = date + '/' + run
        shot_num = str(shot_num)
        ss = 'Shot' + '0'*(3 - len(shot_num)) + shot_num
        file_ext = '.tiff'

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
        raw_img = np.copy(self.raw_img)
        try:
            raw_img = np.array(imread(path), dtype=float)
            if overwrite:
                self.raw_img = np.copy(raw_img)
        except FileNotFoundError as e:
            print('FileNotFoundError caught', e)
        return np.copy(raw_img)
    
    def FFT(self, image, shift=True):
        """returns f_x axis, f_y axis and fourier transform. All 2D
        """
        Z = np.copy(image)
        Z -= np.mean(Z)
        F_Z = np.fft.fft2(Z)
        if shift==True:
            F_Z = np.fft.fftshift(F_Z)
        
        nrows, ncols = F_Z.shape
        u,v = np.arange(ncols) - ncols//2, np.arange(nrows) - nrows//2
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
    
    def Gauss(self, X,Y,ux,uy,ox,oy,gamma_x,gamma_y):
        """Normalised 2D super-gaussian for masking fourier space
        """
        index_x = ((X-ux)**2/(2.*ox**2))**gamma_x
        index_y = ((Y-uy)**2/(2.*oy**2))**gamma_y
        f = np.exp( -(index_x + index_y))
        area = np.trapz(np.trapz(f, X[0]), Y[:,0])
        return f/area

    def get_fringe_image(self, raw_img, overwrite=True):
        """raw image cropped to just where fringes are. also changes fringes_img_extent
        """
        top, bottom, left, right = self.fringes_roi
        #self.fringes_img_extent = [left, right, bottom, top]
        Z = np.copy(raw_img)
        Z = Z[top:bottom, left:right]
        if overwrite:
            self.fringes_img = np.copy(Z)
        return Z
    
    def calc_phase(self, fringes_img, shift_in=None, shift_out=None, ref_raw_imgs=None, overwrite=True):
        """ref_raw_imgs is assumed to be np.array of many reference images

        THIS IS UNLIKE MOST OTHER FUNCTIONS IN CLASS THAT WILL DEFAULT TO ana_settings VALUES
        """
        Z = np.copy(fringes_img)

        if ref_raw_imgs == []:
            # class default, use same protocol as if user specified None
            ref_raw_imgs = None
       
        if ref_raw_imgs is None:
            ref_fringe_imgs = np.array([np.ones_like(fringes_img)])
            
        else:
            ref_raw_imgs = np.copy(ref_raw_imgs)

            if ref_raw_imgs.shape[1:] != Z.shape:
                if ref_raw_imgs.shape[1:] == (492, 656):
                    # roi hasn't been applied yet
                    top, bottom, left, right = self.fringes_roi
                    ref_fringe_imgs = np.array([rz[top:bottom, left:right] for rz in ref_raw_imgs])
                    
                else:
                    print("ref_Z shape doesn't match fringes_img or full image. Ignoring")
                    ref_fringe_imgs = np.array([np.ones_like(fringes_img)])

        if shift_in is None:
            shift_in = self.shift_in
        if shift_out is None:
            shift_out = self.shift_out
 
        _, _, F_Z = self.FFT(Z, shift_in)
        F_Z_masked = F_Z * self.fourier_mask
        _, _, Z = self.iFFT(F_Z_masked, shift_out)

        #ref_fringe_imgs = [self.get_fringe_image(ri, overwrite=False) for ri in ref_raw_imgs] #done above now
        ref_F_Zs = [self.FFT(rfi, shift_in)[-1] for rfi in ref_fringe_imgs]
        ref_F_Z_maskeds = [rfz * self.fourier_mask for rfz in ref_F_Zs]
        ref_Zs = [self.iFFT(rfz, shift_out)[-1] for rfz in ref_F_Z_maskeds]
        ref_Z = np.mean(ref_Zs, axis=0)

        phase = np.angle(Z / ref_Z)
        phase = unwrap_phase(phase)
        
        if overwrite:
            self.phase = np.copy(phase)
        
        return phase
    
    def get_ref_images(self, ref_shots):
        """Given list of them, return array of all images
        """
        ref_filepaths = [self.get_filepath(i) for i in ref_shots]
        ref_raw_imgs = [self.get_raw_image(p, overwrite=False) for p in ref_filepaths]
        return np.array(ref_raw_imgs)

    def check_ref_images(self, fringes_img, shift_in=None, shift_out=None, ref_raw_imgs=None):
        """to help with choosing references - returns lots of things for plotting
        nothing to overwrite
        """
        Z = np.copy(fringes_img)
        if ref_raw_imgs == []:
            # class default, use same protocol as if user specified None
            ref_raw_imgs = None
       
        if ref_raw_imgs is None:
            ref_fringe_imgs = np.array([np.ones_like(fringes_img)])

        else:
            ref_raw_imgs = np.copy(ref_raw_imgs)

            if ref_raw_imgs.shape[1:] != Z.shape:
                if ref_raw_imgs.shape[1:] == (492, 656):
                    # roi hasn't been applied yet
                    top, bottom, left, right = self.fringes_roi
                    ref_fringe_imgs = np.array([rz[top:bottom, left:right] for rz in ref_raw_imgs])
                    
                else:
                    print("ref_Z shape doesn't match fringes_img or full image. Ignoring")
                    ref_fringe_imgs = np.array([np.ones_like(fringes_img)])

        ref_fringes_img = np.mean(ref_fringe_imgs, axis=0)

        if shift_in is None:
            shift_in = self.shift_in
        if shift_out is None:
            shift_out = self.shift_out

        ref_F_Zs = [self.FFT(rfi, shift_in)[-1] for rfi in ref_fringe_imgs]
        ref_F_Z = np.mean(ref_F_Zs, axis=0)
        U, V, _ = self.FFT(ref_fringe_imgs[0], shift_in)

        ref_F_Z_maskeds = [rfz * self.fourier_mask for rfz in ref_F_Zs]
        ref_F_Z_masked = np.mean(ref_F_Z_maskeds, axis=0)
        
        return ref_fringes_img, U, V, ref_F_Z, ref_F_Z_masked
    
    def no_ref_correction(self, overwrite=True):
        """Entered if no ref image provided.
        
        Remove phase gradient due to fringes alone
        """
        
        # correct for fringe gradient first 
        phase_masked = np.copy(self.phase)
        ROI = self.channel_roi
        
        raw = self.raw_img
        nrows, ncols = raw.shape
        blank_out_channel = True
        if np.all(np.array([0, nrows, 0, ncols]) == np.array(ROI)):
            # hasn't been changed since default, don't blank out any of image
            blank_out_channel = False
        
        if blank_out_channel:
            phase_masked[ROI[0]:ROI[1], ROI[2]:ROI[3]] = np.nan
        
        x_trend = np.nanmean(phase_masked, axis=0)
        x_axis = np.arange(x_trend.size)
        polynomials = np.polyfit(x_axis, x_trend, deg=1) #  should just be linear
        poly_x = np.poly1d(polynomials)

        phase_masked -= poly_x(x_axis)[np.newaxis, :] #subsequent y trend already has x trend removed
        
        # correct for image variation 
        y_trend = np.nanmean(phase_masked, axis=1)
        y_axis = np.arange(y_trend.size)
        polynomials = np.polyfit(y_axis, y_trend, deg=3) #  should just be linear
        poly_y = np.poly1d(polynomials)

        phase = np.copy(self.phase)
        phase -= poly_x(x_axis)[np.newaxis, :]
        phase -= poly_y(y_axis)[:, np.newaxis]

        if overwrite:
            self.phase = np.copy(phase)

        return (x_axis, x_trend, poly_x(x_axis), y_axis, y_trend, poly_y(y_axis)), phase
    
    def calc_channel_mask(self, overwrite=True):
        ROI = self.channel_roi
        phase = self.phase 
        
        mask = np.zeros_like(phase)
        mask_extent = np.copy(self.fringes_img_extent)
        
        ch_top, ch_bottom, ch_left, ch_right = ROI
        im_left, im_right, im_bottom, im_top = mask_extent
        
        #mask[ROI[0]:ROI[1], ROI[2]:ROI[3]] = 1.0
        mask[ ch_top-im_top : ch_bottom-im_top, ch_left - im_left : ch_right - im_left] = 1.0
        
        if overwrite:
            self.channel_mask = np.copy(mask)
        return mask
    
    def correct_phase_polarity(self, overwrite=True):
        phase = self.phase
        c_mask = self.channel_mask
        not_c_mask = np.where(c_mask==1.0, 0.0, 1.0)

        # just do in vertical axis as channel more prominent
        inner = np.nanmean(phase * c_mask, axis=1)
        inner = inner[inner != 0.]
        inner = np.mean(inner)
        
        outer = np.nanmean(phase * not_c_mask, axis=1)
        outer = outer[outer != 0.]
        outer = np.mean(outer)
        # old way
        #inner = np.nanmean(phase * c_mask)
        #outer = np.nanmean(phase * not_c_mask)
        
        if inner < outer:
            phase = -1.0 * phase
            
        if overwrite:
            self.phase = np.copy(phase)
        
        return phase
    
    def calc_channel_angle(self, n_std_threshold = None, overwrite=True):
        """Fits 1D line to maxima in channel
        angle returned in deg
        """
        phase = np.copy(self.phase)
        c_mask = np.copy(self.channel_mask)

        if n_std_threshold is None:
            n_std_threshold = self.n_std_threshold
        
        phase = phase * c_mask
        phase -= np.nanmean(phase)
        
        # defaults
        theta,c = np.nan, np.nan
        
        # first guess find maxima of each column
        nrows, ncols = phase.shape
        x = np.arange(ncols)
        yc = np.argmax(phase, axis=0)
        
        # will default to zero where mask made all nans
        ids = yc > 0

        if len(x[ids])==0:
            print('Channel angle fitting only attempted once, n_std threshold (%s) could be too small?' % (n_std_threshold))
            return theta, c, x, yc

        x = x[ids]
        yc = yc[ids]

        # 1D fit the phase maxima
        coeff = np.polyfit(x, yc, deg=1)
        poly = np.poly1d(coeff)

        # check how good fit is and redo
        #residuals = np.abs(yc - poly(x))
        #ids = residuals < n_std_threshold * np.std(residuals)

        # alternate method - clean up fitting to those that mostly resemble linear fit
        residuals = yc - poly(x)
        percent_frac = n_std_threshold * 100.0
        lower, upper = np.percentile(residuals, 100.0 - percent_frac), np.percentile(residuals, percent_frac)
        ids = (residuals >= lower) & (residuals <= upper)
        
        if len(x[ids])==0:
            print('Channel angle fitting only attempted once, n_std threshold (%s) could be too small' % (n_std_threshold))
            
            return theta, c, x, yc

        x = x[ids]
        yc = yc[ids]
        coeff = np.polyfit(x, yc, deg=1)
        poly = np.poly1d(coeff)
        
        m,c = coeff
        theta = np.arctan(m) * 180.0/np.pi
		
        if overwrite:
            self.channel_angle = np.copy(theta)
            self.channel_offset = np.copy(c)
        
        return theta, c, x, yc

    def rotate_image(self, image, theta=0.0):
        """theta in deg
        """
        if np.isnan(theta):
            theta = 0.0
        im = np.copy(image)
        im = rotate(im, theta)
        return im


    def calc_phase_centre(self, method=None, axes=(0,1), overwrite=True):
        """finds phase central axis for abel inversion based on just the channel.
        centre will be for an image with the channel verticla (what PyAbel wants)
        
        method kwarg passed onto PyAbel's find_origin function.
        If given as a list, then it will do this method to each axis.
        axes is 0, 1 or (0,1) for centering. 0 is horizontal, 1 is vertical
        """
        if method is None:
            method = self.centre_method
            
        phase = np.copy(self.phase)
        c_mask = np.copy(self.channel_mask)
        phase_roi = phase * c_mask

        phase = phase.T #abel wants the inversion axis to be vertical
        phase_roi = phase_roi.T
        
        if type(method)==list:
            
            if self.abel_version == '0.8.4':
                # original done in Mirage
                centre_x, _ = abel.tools.center.find_origin(phase_roi, method=method[0], axes=0)
                _, centre_y = abel.tools.center.find_origin(phase_roi, method=method[1], axes=1)
            
            else:
                # old done using version 0.8.3
                #centre_x, centre_y = abel.tools.center.find_center_by_convolution(phase_roi)
                centre_x, _ = abel.tools.center.find_center_by_convolution(phase_roi)
                # looks to be a - b here? maybe just a fluke
                _, centre_y = abel.tools.center.find_center_by_gaussian_fit(phase_roi)
                
            centre = (centre_x, centre_y)
            
        else:
            centre = abel.tools.center.find_center(phase_roi, method=method, axes=axes)
        
        
        if overwrite:
            self.centre = centre
            
        return centre

    def phase_to_SI(self, overwrite=True):
        """converts captured phase to integrated n_e (cm-3) along line of sight
        """
        phase = np.copy(self.phase)
        mperpixel = self.umperpixel * 1e-6
        int_ne_dl_SI = phase * self.m_e * 1e7 / (self.e**2 * self.lambda_0 ) / mperpixel
        int_ne_dl = int_ne_dl_SI * (1e-6) # per cm3
        
        if overwrite:
            self.int_ne_dl = np.copy(int_ne_dl)
        
        return int_ne_dl
        
    
    def abel_invert_int_ne_dl(self, method=None, overwrite=True):
        """method kwarg passed on to PyAbel's Transform function
        """
        if method is None:
            method = self.inversion_method
        int_ne_dl = np.copy(self.int_ne_dl)
        
        nrows, ncols = int_ne_dl.shape
        int_ne_dl = int_ne_dl.T # PyAbel wants the inversio nais vertical
        centre = np.copy(self.centre)
        
        if self.abel_version == '0.8.4': 
            centred_image = abel.tools.center.set_center(int_ne_dl, origin=centre, axes=(0,1))
        else:
            #centre = (centre[1], centre[0]) # annoying correction in using OLD abel - this correction seems wrong?
            centred_image = abel.tools.center.set_center(int_ne_dl, center=centre)#, axes=1)
            
        
        if method == 'direct':
            
            # stop default to cython and speed up calculation
            t_o = {'backend': 'python'}
            ne = abel.Transform(centred_image, method=method, 
                                             direction='inverse', transform_options=t_o).transform
        
        else:
            ne = abel.Transform(centred_image, method=method, 
                                             direction='inverse').transform
        
        ne = ne.T
        
        ne[ne==0.0] = np.nan
        
        if overwrite:
            self.ne = np.copy(ne)
            
        return ne
    
    
    def get_ne(self, path):
        # do all processing in one func
        self.reset()
        
        self.get_raw_image(path)
        
        if np.all(np.isnan(self.raw_img)):
            # no file found - ignore the rest
            return np.copy(self.ne.T)
            
        self.get_fringe_image(self.raw_img)
        
        self.calc_phase(self.fringes_img, ref_raw_imgs=self.ref_raw_imgs)

        self.calc_channel_mask()
        if self.apply_no_ref_correction:
            self.no_ref_correction()
        
        if self.invert_phase==True:
            # override attempt if unreliable
            phase = np.copy(self.phase)
            phase = -1.0 * phase
            self.phase = np.copy(phase)
        else:
            self.correct_phase_polarity()
        
        channel_theta, channel_offset, _, _ = self.calc_channel_angle()
        self.phase = self.rotate_image(self.phase, channel_theta)
        
        if self.fixed_channel_centre != False:
            self.centre = self.fixed_channel_centre
        else:
            self.calc_phase_centre()
        self.phase_to_SI()
        self.abel_invert_int_ne_dl()
        
        return np.copy(self.ne.T)
    

    def get_ne_from_img(self, img):
        """For use with data Pipeline
        """
        self.reset()
        
        self.raw_img = np.array(img, dtype=float)
        
        if np.all(np.isnan(self.raw_img)):
            # no file found - ignore the rest
            return np.copy(self.ne.T)
            
        self.get_fringe_image(self.raw_img)
        
        self.calc_phase(self.fringes_img, ref_raw_imgs=self.ref_raw_imgs)

        self.calc_channel_mask()
        if self.apply_no_ref_correction:
            self.no_ref_correction()
        
        if self.invert_phase==True:
            # override attempt if unreliable
            phase = np.copy(self.phase)
            phase = -1.0 * phase
            self.phase = np.copy(phase)
        else:
            self.correct_phase_polarity()
        
        channel_theta, channel_offset, _, _ = self.calc_channel_angle()
        self.phase = self.rotate_image(self.phase, channel_theta)

        centre = self.calc_phase_centre()
        self.phase_to_SI()
        self.abel_invert_int_ne_dl()
        
        return np.copy(self.ne.T)
        
    
    def get_channel_info(self, path):
        """Only do up to the channel angle retrieval
        """
        self.reset()
        
        self.get_raw_image(path)
        
        if np.all(np.isnan(self.raw_img)):
            # no file found - ignore the rest
            return np.copy(self.channel_angle), np.copy(self.channel_offset)
            
        self.get_fringe_image(self.raw_img)
        
        self.calc_phase(self.fringes_img, ref_raw_imgs=self.ref_raw_imgs)

        self.calc_channel_mask()
        if self.apply_no_ref_correction:
            self.no_ref_correction()
        
        if self.invert_phase==True:
            # override attempt if unreliable
            phase = np.copy(self.phase)
            phase = -1.0 * phase
            self.phase = np.copy(phase)
        else:
            self.correct_phase_polarity()
        
        channel_theta, channel_offset, _, _ = self.calc_channel_angle()
        
        return np.copy(self.channel_angle), np.copy(self.channel_offset)
        
        
    
    def get_ne_lineout(self, path):
        """Return 1d average lineout of density
        """
        self.reset()
        
        self.get_ne(path)
        
        ne = np.copy(self.ne)
        channel_width = np.copy(self.channel_width)
        nrows, ncols = ne.shape
        mid = nrows // 2
        hw = channel_width // 2
        
        avg = np.nanmean(ne[mid - hw: mid + hw, :], axis=0)
        top = np.nanmean(ne[mid - hw: mid, :], axis=0)
        bottom = np.nanmean(ne[mid: mid+hw, :], axis=0)
        
        #ne_lineout = np.nanmean( ne[mid-hw:mid+hw, :], axis=0)

        return np.array([avg, top, bottom])
    
    
    def get_ne_lineout_from_img(self, img):
        """Return 1d average lineout of density
        """
        self.reset()
        
        self.get_ne_from_img(img)
        
        ne = np.copy(self.ne)
        channel_width = np.copy(self.channel_width)
        nrows, ncols = ne.shape
        mid = nrows // 2
        hw = channel_width // 2
        
        avg = np.nanmean(ne[mid - hw: mid + hw, :], axis=0)
        top = np.nanmean(ne[mid - hw: mid, :], axis=0)
        bottom = np.nanmean(ne[mid: mid+hw, :], axis=0)
        
        #ne_lineout = np.nanmean( ne[mid-hw:mid+hw, :], axis=0)

        return np.array([avg, top, bottom])
    
    def get_img_from_img(self, img):
        """Test for DataPipeline
        """
        self.reset()
        
        return img
    
    
    def get_phase_map(self, path):
        """just return phase map instead of raw_img. Will be cropped to 
        fringes region
        """
        self.reset()
        
        self.get_raw_image(path)
        
        if np.all(np.isnan(self.raw_img)):
            # no file found - ignore the rest
            return np.copy(self.ne.T)
            
        self.get_fringe_image(self.raw_img)
        
        self.calc_phase(self.fringes_img, ref_raw_imgs=self.ref_raw_imgs)

        self.calc_channel_mask()
        if self.apply_no_ref_correction:
            self.no_ref_correction()
        
        if self.invert_phase==True:
            # override attempt if unreliable
            phase = np.copy(self.phase)
            phase = -1.0 * phase
            self.phase = np.copy(phase)
        else:
            self.correct_phase_polarity()
        
        return self.phase.T


    def Gauss_1D(self, x, mu, o, A, c):
            return A * np.exp( - 0.5 * (x-mu)**2 / o**2) + c
    
    def get_guass_fit_ne(self, path, plotter=True):
        """Fits ne to guass in vertical axis. 
        
        ne is cropped to only include channel and averaged over whole channel
        to give size in y
        """
        
        self.reset()
        self.get_ne(path)
        ne_crop = np.copy(self.ne)
        
        x = self.ne_y_mm
        
        ne_plateau = np.nanpercentile(ne_crop, 95)
        ne_crop[ne_crop < 0.05 * ne_plateau] = np.nan
        y = np.nanmean(ne_crop,axis=1)
        
        # could be all nans if no image found
        if np.all(np.isnan(y)):
            popt = [np.nan]*4
            perr = [np.nan]*4
            return popt, perr     
        
        # give some initial guesses
        c = np.nanmin(y)
        A = np.nanmax(y) - c
        mu = np.nansum(x * y) / np.nansum(y)
        o = (np.nansum(y * (x-mu)**2)/np.nansum(y))**(0.5)
        p0 = [mu, o, A, c]
        ids = np.isfinite(y)
     
        popt, pcov = curve_fit(self.Gauss_1D, x[ids], y[ids], p0)
        perr = np.diagonal(pcov)**(0.5)
                      
        if plotter == True:
            plt.figure()
            plt.imshow(ne_crop), plt.title('Cropped $n_e$')
            
            plt.figure()
            plt.plot(x,y)
            plt.title('Fitted vertical channel')
            dx = np.linspace(np.min(x), np.max(x), 100)
            plt.plot(dx, self.Gauss_1D(dx, *popt), 'r--')
        
        return popt, perr        
        
        


    def save_ana_settings(self):
        filedir = self.cal_data_path + '/' + self.diag + '/'
        date, run = self.run_name.split('/')
        shot_num = str(self.shot_num)
        ss = 'shot' + '0'*(3 - len(shot_num)) + shot_num #LOWERCASE s

        filename = self.diag + '_ana_settings_' + date + '_' + run + '_' + ss + '.pkl'

        dicto = {}
        list_of_ana_settings = ['umperpixel', 'fringes_roi', 'fourier_mask',
                                'channel_roi', 'channel_width', 'ref_shots', 'apply_no_ref_correction',
                                'centre_method', 'inversion_method', 'shift_in', 'shift_out', 'n_std_threshold',
                                'invert_phase', 'fixed_channel_centre']
        
        for key in list_of_ana_settings:
            if hasattr(self, key):
                dicto[key] = getattr(self, key)
        
        save_object(dicto, filedir+filename)
        print('written: ', filedir+filename)
