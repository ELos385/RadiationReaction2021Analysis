import numpy as np
from skimage.io import imread
from skimage.restoration import unwrap_phase
from scipy.signal import find_peaks
from scipy.constants import c as lightc
from scipy.interpolate import interp1d

from config import *

class Timing:
    """

    """
    def __init__(self,CALIBRATION_TO_LOAD=HOME+'/calib/Timing/' + "2019_la3_calibration.txt"):
        """ Settings needed for analysis, as set by Peter
        """
        
        self.CALIBRATION_TO_LOAD = CALIBRATION_TO_LOAD
        
        self.INTERVALS = 2 ** 11  # number of frequency intervals to interpolate over, for f.t.
        self.CROP_TOP = 380  # top of map of phases to average to
        self.CROP_BOTTOM = 580  # bottom of map of phases to average to
        self.FRINGES_LEFT = 950  # leftmost pixel for linear fit
        self.FRINGES_RIGHT = 1650  # rightmost pixel for linear fit
        self.MASK_FACTOR = 3 # exlude values of noise if intensity < than MASK_FACTOR * median intensity

        self.DETECTOR_SIZE = 2750
        
        # standard for which Andor cam?
        self.nrows = 1000
        self.ncols = 2750
        
    # interpolates data to a linear frequency scale
    def interpolate(self, frequencies, intensity_data):
        freq_intervals = np.linspace(min(frequencies),max(frequencies),self.INTERVALS)
        fnc = interp1d(frequencies,intensity_data, axis = 1)
        interpolated_intensities = fnc(freq_intervals)
        return freq_intervals, interpolated_intensities

    #applies bandpass filer 5 intervals either side of left peak
    def apply_bandpass(self, data):
        #skips the edges as it's inconsistant if this is seen as a peak
        peak_prominance = 0.6*max(np.sum(np.abs(data[20:data.shape[0]-20]),axis = 1) )
        peaks = find_peaks(np.sum(np.abs(data[20:data.shape[0]-20]),axis = 1), prominence=peak_prominance,distance=10)
        peaks = peaks[0]
        peaks = peaks + 20
        bandpass = np.zeros(np.shape(data))
        bandpass[peaks[1]-15:peaks[1]+13,:] = 1
        data = data*bandpass
        return data


    # get calibration data
    def load_calibration_data(self):
        with open(self.CALIBRATION_TO_LOAD) as file:
            calibration_coeffs_text = file.read().strip().split(",")

        calibration_coeffs_num = [float(txt) for txt in calibration_coeffs_text]

        # enumerate polynomial
        calibration_coeffs_num.reverse()
        wavelengths = np.polyval(calibration_coeffs_num, np.arange(1, self.DETECTOR_SIZE + 1))
        frequencies = lightc / 1e6 / wavelengths[:self.DETECTOR_SIZE]
        return frequencies


    def get_delay_from_path(self, IMAGE_TO_LOAD):
        """main function called to get delay from LivePLotting
        
        renamed to get_delay_from_path from get_delay_from_IMAGE
        
        IMAGE_TO_LOAD is the path
        """
    
        #load image
        img = imread(IMAGE_TO_LOAD)
        img = img[:,:2750]

        #intepolate
        frequencies = self.load_calibration_data()
        freq_interpolated, intensity_interpolated = self.interpolate(frequencies[:img.shape[1]],img)

        #fourier transform
        img_fft_vertical = np.fft.fft(intensity_interpolated, axis=0)
        #apply bandpass filter
        try:
            img_fft_vertical = self.apply_bandpass(img_fft_vertical)
        except:
            print("Bad Image")

        sideband = np.fft.ifft(img_fft_vertical, axis=0)
        phases = np.angle(sideband)
        amplitudes = np.abs(sideband)


        #unwrap phase
        noise_threshold = np.median(amplitudes*self.MASK_FACTOR)
        phase_unwrapped = unwrap_phase(np.ma.masked_where(amplitudes < noise_threshold, phases))
        mean_unwrapped_phases = np.mean(phase_unwrapped[self.CROP_TOP:self.CROP_BOTTOM], axis = 0)



        #calculate delay = 2pi/m
        left_limit = int(self.FRINGES_LEFT*(self.INTERVALS/img.shape[1]))
        right_limit = int(self.FRINGES_RIGHT*(self.INTERVALS/img.shape[1]))
        gradient = np.polyfit(freq_interpolated[left_limit:right_limit], mean_unwrapped_phases[left_limit:right_limit], 1)

        delay = gradient[0]/(2*np.pi)
        
        self.delay = delay
        
        return delay
    
    
    def get_delay_from_img(self, img):
        """main function called to get delay
        
        im is the image already grabbed (either by get_raw_img or from the pipeline)
        """
    
        #load image
        img = img[:,:2750]

        #intepolate
        frequencies = self.load_calibration_data()
        freq_interpolated, intensity_interpolated = self.interpolate(frequencies[:img.shape[1]],img)

        #fourier transform
        img_fft_vertical = np.fft.fft(intensity_interpolated, axis=0)
        #apply bandpass filter
        try:
            img_fft_vertical = self.apply_bandpass(img_fft_vertical)
        except:
            print("Bad Image")

        sideband = np.fft.ifft(img_fft_vertical, axis=0)
        phases = np.angle(sideband)
        amplitudes = np.abs(sideband)


        #unwrap phase
        noise_threshold = np.median(amplitudes*self.MASK_FACTOR)
        phase_unwrapped = unwrap_phase(np.ma.masked_where(amplitudes < noise_threshold, phases))
        mean_unwrapped_phases = np.mean(phase_unwrapped[self.CROP_TOP:self.CROP_BOTTOM], axis = 0)



        #calculate delay = 2pi/m
        left_limit = int(self.FRINGES_LEFT*(self.INTERVALS/img.shape[1]))
        right_limit = int(self.FRINGES_RIGHT*(self.INTERVALS/img.shape[1]))
        gradient = np.polyfit(freq_interpolated[left_limit:right_limit], mean_unwrapped_phases[left_limit:right_limit], 1)

        delay = gradient[0]/(2*np.pi)
        
        self.delay = delay
        
        return delay
    
    def get_raw_img(self, path):
        """main function call for liveplotting - see a given shot
        """
        raw_img = np.full((self.nrows, self.ncols), np.nan)
        
        try:
            raw_img = np.array(imread(path), dtype=float)
        except FileNotFoundError as e:
            print('FileNotFoundError caught', e)
            
        return np.copy(raw_img)