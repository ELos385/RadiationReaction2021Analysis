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
    img = get_signal_region(im, n_sigma, kernel_size)
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

def get_shadows_from_clicks(x, y, list_of_rough_peaks, 
                            wire_width=2.8, smoothing_factor=1e-5, 
                            plotting=True):
    """basically same workflow as find_wire_shadows just without hte auto-finding 
    shadows that always seemed to fail!
    
    Instead give a list of rough locations of shadows as a starting point
    
    wire_width now in units of x [mm]
    """
    peaks = np.copy(list_of_rough_peaks)
    peak_idxs = []
    for p in peaks:
        p_idx = np.argmin((x - p)**2)
        peak_idxs.append(p_idx)    
    
    # get underlying signal without wires    
    y_no_wires = np.copy(y)
    for p in peak_idxs:
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
        plt.plot(x, y, label='Given signal')
        plt.plot(x[peak_idxs], y[peak_idxs], "x", label='Shadow peaks')
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
    goodness = []
    lines = []
    for idx, p in enumerate(peak_idxs):
        lb,ub = x[p] - 2*wire_width, x[p] + 2*wire_width
        ids = (x >= lb) & (x <=  ub)
        
        x_d = x[ids]
        diff_d = diff[ids]
        
        p0 = [x[p], wire_width/2.0, 0.1, 0.0]
        # G(x,mu,o,A,c)
        # mu, o, A, c
        #bounds = ((lb, ub), (0, 2.0), (0, 2.0), (-0.1, +0.1))
        bounds = ((lb, ub), (1e-1, 4.0), (0.0, 1.0), (-0.1, +0.1))
        bounds = tuple(zip(*bounds))
        
        try:
            popt, pcov = curve_fit(G, x_d, diff_d, p0, bounds=bounds)
            perr = np.diag(pcov)**(0.5)
            # print(popt)
            
            # check fit around peak
            x_ids = (x_d >= popt[0]-popt[1]) & (x_d <= popt[0]+popt[1])
            goodness_check = np.mean( (G(x_d[x_ids], *popt) - diff_d[x_ids])**2 )
            goodness.append(goodness_check)
            
            if plotting == True:
                if idx==0:
                    line, = plt.plot(x_d, G(x_d, *popt), 'r', ls='-.', label='Gaussian fit')
                else:
                    line, = plt.plot(x_d, G(x_d, *popt), 'r', ls='-.')
                lines.append(line)    
                
        except(RuntimeError):
            # couldn't fit - just ignore it
            popt = np.full_like(p0, fill_value=np.nan)
            perr = np.full_like(p0, fill_value=np.nan)
        shadows.append([popt[0], perr[0], popt[1], perr[1]])
        
    if plotting == True:
        plt.legend()
    
    shadows = np.array(shadows)
    
    # final removal if too many shadows tried to be fitted
    g = goodness
    print(np.abs(g - np.median(g))/np.std(g))
    
    n_sigma = 4.2
    non_valid = goodness > np.nanmedian(goodness) + n_sigma * np.nanstd(goodness)
    
    shadows = shadows[~non_valid,:]
    if plotting==True:
        lines = np.array(lines)
        to_remove = lines[non_valid]
        for l in to_remove:
            l.set_color('g')
    
    
    return shadows






# wire_shadow_idx (0-19), lb [mm], ub [mm]
_espec1wire_bounds = [[5, 85, 95],
          [6, 95, 105],
          [7, 110,115],
          [8, 118, 123],
          [9, 123, 126],
          [10, 135, 140],
          [11, 147, 152],
          [12, 160, 165],
          [13, 165, 170], 
          [14, 173, 178],
          [15, 185, 190],
          [16, 198, 203],
          [17, 210, 215],
          [18, 218, 223],
          [19, 223, 228]
        ]

_espec2wire_bounds = [[5, 60, 70],
          [6, 80, 90],
          [7, 97,106],
          [8, 107, 115],
          [9, 117, 125],
          [10, 135, 145],
          [11, 155, 165],
          [12, 170, 180],
          [13, 182, 190], 
          [14, 190, 200],
          [15, 210, 217],
          [16, 228, 238],
          [17, 247, 255],
          [18, 256, 265],
          [19, 266, 275]
        ]

def normal(X, mu, fwhm):
    sigma = fwhm / 2.355
    pre = sigma * (2.0*np.pi)**(0.5)
    index = (X-mu)**2 / (2.0 * sigma**2)
    return pre * np.exp(-index)



def normalise(z, x, y=None):
    """Normalise a 2D or 1D distribution
    """
    s = np.trapz(z, x)
    if y is not None:
        s = np.trapz(s, y)
    s = np.abs(s)
    return z.copy()/s

def find_FWHM(x,y,frac=0.5):
    """Brute force FWHM calculation.
    Frac allows you to easily change, so to e-2 value etc.
    
    """
    fm = y.copy() - y.min()
    fm = fm.max()
    hm = fm * frac

    hm += y.min()
    fm = fm + y.min()
    max_idx = np.argmax(y)
    
    first_half = np.arange( int(0.9 * max_idx) )
    second_half = np.arange( int(1.1 * max_idx), x.size )
    
    hm_x_left = np.interp(hm, y[first_half], x[first_half])
    hm_x_right = np.interp(hm, y[second_half][::-1], x[second_half][::-1])
    
    return hm_x_right - hm_x_left


def get_full_shadows(s,s_err, bounds_array_name='espec1'):
    """Given list of shadows and errors, function returns them in an array
    of size = 20, for all the wires used on experiment, with their values in the
    right position.
    
    list idx 0 is the high energy end of the shadows, idx -1 is the low energy end.
    
    now bounds_array_name just diag name
    bounds array should be np.array of shape(N,3) for N wires, where 
    col 0 is shadow id (0 being highest energy, 19 being lowest)
    col 1 is lower bound of where it could be on screen [m]
    col 2 is upper bound [m]
    """
    if bounds_array_name=='espec1':
        bounds_array = np.copy(_espec1wire_bounds) * [1, 1e-3, 1e-3]
    elif bounds_array_name=='espec2':
        bounds_array = np.copy(_espec2wire_bounds) * [1, 1e-3, 1e-3]
    else:
        print('Incorrect diag name')
        return np.nan, np.nan
    
    new_shadows = np.full((20), fill_value=np.nan)
    new_shadows_err = np.full((20), fill_value=np.nan)
    for i,j in zip(s, s_err):
        ids = (i >= bounds_array[:,1]) & (i <= bounds_array[:,2])
        idx = int(bounds_array[ids,0][0])
        new_shadows[idx] = i
        new_shadows_err[idx] = j
    return new_shadows, new_shadows_err


from scipy.interpolate import RectBivariateSpline

def find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=1e-5,
                             plotter=True):
    """Given electron tracking maps, function returns the parameter space 
    (E_min, E_max, theta_min, theta_max) range that the inner product of the 
    shadow measurements, s1 ± ds1, s2 ± ds2, cover.
    
    Range is defined as the everything >= (lim * peak value) of the 
    normalised inner product.
    """
    # get overlap region
    X1, X2 = np.copy(X1), np.copy(X2)
    t_axis, p_axis = np.copy(t_axis), np.copy(p_axis)
    
    T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    X1 *= normal(X1, s1, ds1)
    X2 *= normal(X2, s2, ds2)
    X_prod = X1 * X2
    X_prod = normalise(X_prod, t_axis, p_axis)
    X_sum = X1 + X2
    
    if plotter == True:
        plt.figure()
        cax = plt.pcolormesh(T_axis, P_axis, X_prod)
        plt.colorbar(cax)
        plt.title('$X_{1} \cdot X_{2}$')
        
        plt.figure()
        cax = plt.pcolormesh(T_axis, P_axis, X_sum)
        plt.colorbar(cax)
        plt.title('$X_{1} + X_{2}$')
    
    # find parameter space the overlap occupies 
    # roi in parameter space is min and max where 
    # X_prod >= X_prod * lim 
    lineout1 = np.nanmean(X_prod, axis=0)
    lineout1_lim = np.nanmax(lineout1)*lim
    t_valid = t_axis[lineout1 >= lineout1_lim]
    if t_valid.size == 0:
        print('Disjoint PDFs')
        tl, tr= np.nan, np.nan
    else:
        tl, tr = t_valid[0], t_valid[-1]
    t_peak = t_axis[np.argmax(lineout1)]
    
    lineout2 = np.nanmean(X_prod, axis=1)
    lineout2_lim = np.nanmax(lineout2)*lim
    p_valid = p_axis[lineout2 >= lineout2_lim]
    if p_valid.size == 0:
        print('Disjoint PDFs')
        pl, pr = np.nan, np.nan
    else:
        pl, pr = p_valid[0], p_valid[-1]
    p_peak = p_axis[np.argmax(lineout2)]

    if plotter == True:
        plt.figure()    
        plt.plot(t_axis, lineout1, color=colors[0])
        plt.axhline(y=lineout1_lim, color=colors[0])    
        plt.axvline(x=tl, color=colors[0], ls='--')
        plt.axvline(x=tr, color=colors[0], ls='--')
        
        plt.plot(p_axis, lineout2, color=colors[1])
        plt.axhline(y=lineout2_lim, color=colors[1])
        plt.axvline(x=pl, color=colors[1], ls='--')
        plt.axvline(x=pr, color=colors[1], ls='--')
        plt.title('sub-region')
    
    if tr<tl: tl,tr=tr,tl
    if pr<pl: pl,pr=pr,pl
    
    return t_peak, p_peak, tl, tr, pl, pr


def find_wire_shadow_param_space(X1, X2, t_axis, p_axis, 
                                 s1, s1_err, s2, s2_err,
                                 number_of_iterations=4,
                                 ds_factor_start=1000, ds_decrease_factor_on_iter=10, refine_factor_on_iter=10,
                                 lim=1e-2, plotter=True):
    """iterates find_overlap_param_space a number of times. On each iteration 
    the parameter space is restricted to lim * peak, and the artificial 
    enhancing of the error on s1, s2 is decreased by ds_decrease_factor_on_iter
    and the resolution in the sample area is increased by refine_factor_on_iter.
    
    Returns t_peak, t_width, p_peak, p_width
    
    """
    T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    for i in range(1, number_of_iterations+1):
        ds_factor = ds_factor_start / (ds_decrease_factor_on_iter)**(i-1) # start high
        ds1 = s1_err * ds_factor
        ds2 = s2_err * ds_factor
        #print('Iteration %s. ds_factor = %s' % (i,ds_factor))
        
        t_peak, p_peak, tl, tr, pl, pr = find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=lim, plotter=plotter)
        
        check = np.array([t_peak, p_peak, tl, tr, pl, pr])
        if np.any(np.isnan(check)):
            return tuple([np.nan]*4)
        
        upsample_factor = refine_factor_on_iter
        
        # repeat but upsampled, in the vicinity of tl-tr, pl-pr
        new_t_axis = t_axis[ (t_axis>=tl) & (t_axis<=tr) ]
        new_t_axis = np.linspace(new_t_axis.min(), new_t_axis.max(), upsample_factor * new_t_axis.size)
        new_p_axis = p_axis[ (p_axis>=pl) & (p_axis<=pr) ]
        new_p_axis = np.linspace(new_p_axis.min(), new_p_axis.max(), upsample_factor * new_p_axis.size)
        new_P_axis, new_T_axis = np.meshgrid(new_p_axis, new_t_axis)
        
        # null - just refines what's already there
        #new_t_axis = np.linspace(t_axis.min(), t_axis.max(), upsample_factor*t_axis.size)
        #new_p_axis = np.linspace(p_axis.min(), p_axis.max(), upsample_factor*p_axis.size)
        
        fZ1 = RectBivariateSpline(P_axis[:,0], T_axis[0,:], X1) 
        X1_upsampled = fZ1(new_p_axis, new_t_axis)
        
        fZ2 = RectBivariateSpline(P_axis[:,0], T_axis[0,:], X2)
        X2_upsampled = fZ2(new_p_axis, new_t_axis)
    
        # to pass onto next iteration
        X1, X2, t_axis, p_axis = X1_upsampled, X2_upsampled, new_t_axis, new_p_axis
        T_axis, P_axis = np.meshgrid(t_axis, p_axis)
    
    # do final time
    ds1 = s1_err 
    ds2 = s2_err
    t_peak, p_peak, tl, tr, pl, pr = find_overlap_param_space(X1, X2, t_axis, p_axis, s1, s2, ds1, ds2, lim=0.5, plotter=plotter)
    
    t_width = 0.5 * (tr - tl)
    p_width = 0.5 * (pr - pl)
    
    return t_peak, t_width, p_peak, p_width