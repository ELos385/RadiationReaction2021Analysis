#espec_wire_tracking.py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.general_tools import *
from lib.folder_tools import *


from scipy.signal import medfilt2d
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def G(x,mu,o,A,c):
    """ generic 1D gaussian func
    """
    return A * np.exp( -0.5 * (x-mu)**2/o**2) + c


def get_signal_region(im, n_sigma=1.0, kernel_size=5):
    """returns im but with 0 for everywhere not identified as useful signal.
    """
    im  = np.copy(im)
    nrows, ncols = im.shape
    x,y = np.arange(ncols), np.arange(nrows)
    X,Y = np.meshgrid(x,y)
    
    im -= np.median(im)
    
    # location where signal is bright in image
    sig_bool = im > n_sigma * np.std(im)
    
    #filter to remove any hard hits detected to just get signal
    sig_bool = medfilt2d(np.array(sig_bool, dtype=np.uint8), kernel_size=kernel_size)
    
    return im * sig_bool



def find_wire_shadows(x_mm, im, plotting = True,
          n_sigma=1.0,
          kernel_size=5,
          prominence=2e-2,
          distance=40,
          wire_width=5,
          smoothing_factor=0.1):
    """
    Function processes image to give wire shadow positions on screen and their 
    widths (sigma of gaussian fit to shadow).
    
    returns (shadow_mid_point, shadow_mid_point_err, shadow_width (std), shadow_width_err)
    
    x_mm shoud be x_axis for image (1d numpy array)
    im should transformed raw image from espec object
    plotting is bool to make figs to show how analysis is working
    
    the rest of the analysis_kwargs control how sensitive function is to help 
    find peaks.
    
    """
    x = np.copy(x_mm)
    
    # get signal region in image
    img = get_signal_region(im)
    y = np.nansum(img, axis=0)
    y /= y.max()
    
    # reverse as useful later
    if np.all(np.diff(x)<0):
        # if descending, reverse for spline fitting
        x = x[::-1]
        y = y[::-1]
        
    # find drops in signal due to wire shadows
    peaks, _ = find_peaks(-y, prominence=prominence, distance=distance, width=wire_width)

    # get underlying signal without wires    
    y_no_wires = np.copy(y)
    for p in peaks:
        lb,ub = x[p] - wire_width/2.0, x[p] + wire_width/2.0
        ids = (x >= lb) & (x <=  ub)
        y_no_wires[ids] = np.nan
    
    # fit y_no_wires smoothly across shadows, so shadow widths can be found
    ids = ~np.isnan(y_no_wires)
    spl = UnivariateSpline(x[ids], y[ids])
    spl.set_smoothing_factor(smoothing_factor)
    
    if plotting==True:
        plt.figure()
        plt.title('Shadow finding analysis')
        plt.plot(x, y/y.max(), label='Given signal')
        plt.plot(x[peaks], y[peaks], "x", label='Shadow peaks')
        plt.plot(x, y_no_wires, label='Signal only')
        plt.plot(x, spl(x), label='Spline fit signal only')
        plt.legend()
        plt.grid()
    

    diff = spl(x) - y
    
    if plotting==True:
        plt.figure()
        plt.title('Shadow finding analysis')
        plt.plot(x, diff, label='Signal - Spline fit')
    
    shadows = []
    for idx, p in enumerate(peaks):
        lb,ub = x[p] - 1.25*wire_width, x[p] + 1.25*wire_width
        ids = (x >= lb) & (x <=  ub)
        
        x_d = x[ids]
        diff_d = diff[ids]
        
        p0 = [x[p], wire_width/2.0, 0.1, 0.0]
        bounds = ((lb, ub), (0, wire_width), (0, np.inf), (-0.1, +0.1))
        
        bounds = tuple(zip(*bounds))
        try:
            popt, pcov = curve_fit(G, x_d, diff_d, p0, bounds=bounds)
            perr = np.diag(pcov)**(0.5)
            # print(popt)
            
            if plotting == True:
                if idx==0:
                    plt.plot(x_d, G(x_d, *popt), 'r', ls='-.', label='Gaussian fit')
                else:
                    plt.plot(x_d, G(x_d, *popt), 'r', ls='-.')
                    
        except(RuntimeError):
            # couldn't fit - just ignore it
            popt = np.full_like(p0, fill_value=np.nan)
            perr = np.full_like(p0, fill_value=np.nan)
        
        shadows.append([popt[0], perr[0], popt[1], perr[1]])
        
    if plotting == True:
        plt.legend()
    
    shadows = np.array(shadows)
    return shadows