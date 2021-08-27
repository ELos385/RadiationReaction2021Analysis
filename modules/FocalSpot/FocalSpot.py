import numpy as np
import scipy.optimize as opt

def two_d_gaussian(T, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x=T[0]
    y=T[1]
    g = amplitude*np.exp(-calc_ellipse(x, y, xo, yo, sigma_x, sigma_y, theta)/2.0)+offset
    if np.sum(T-g)<=0:
        return 10**10
    else:
        return g.ravel()

def calc_ellipse(x, y, xo, yo, sigma_x, sigma_y, theta):
    a = (np.cos(theta)**2)/(sigma_x**2) + (np.sin(theta)**2)/(sigma_y**2)
    b = 2.0*np.cos(theta)*np.sin(theta)*(1.0/(sigma_x**2)-1.0/(sigma_y**2))
    c = (np.sin(theta)**2)/(sigma_x**2) + (np.cos(theta)**2)/(sigma_y**2)
    g = a*((x-xo)**2) + b*(x-xo)*(y-yo) + c*((y-yo)**2)
    return g

def calc_moments_spot(img_array_2D, axs, peak_amp, axs_index):
    """

    """
    im_summed_ax=np.sum(img_array_2D, axis=axs_index)
    im_summed_ax[np.where(im_summed_ax<0.36*peak_amp)]=0.0
    central_pos=np.trapz(im_summed_ax*axs, axs)/np.trapz(im_summed_ax, axs)
    sigma_ax=(np.trapz(im_summed_ax*axs**2, axs)/np.trapz(im_summed_ax, axs)-central_pos**2)**0.5
    return central_pos, sigma_ax


def convert_width_to_FHWM(width):
    return width*(2.0*np.log(2.0))**0.5

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class FocalSpot:
    """

    """
#amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    def __init__(self, focal_pos_x=None, focal_pos_x_err=None, focal_pos_y=None, focal_pos_y_err=None, focal_pos_z=None, focal_pos_z_err=None, FWHM_x=None, FWHM_x_err=None, FWHM_y=None, FWHM_y_err=None, angle_rot=None, angle_rot_err=None, energy_frac_FWHM=None, energy_frac_FWHM_err=None, microns_per_pixel=None, microns_per_pixel_err=None):
        self.focal_pos_x=focal_pos_x
        self.focal_pos_x_err=focal_pos_x_err
        self.focal_pos_y=focal_pos_y
        self.focal_pos_y_err=focal_pos_y_err
        self.focal_pos_z=focal_pos_z
        self.focal_pos_z_err=focal_pos_z_err
        self.FWHM_x=FWHM_x
        self.FWHM_x_err=FWHM_x_err
        self.FWHM_y=FWHM_y
        self.FWHM_y_err=FWHM_y_err
        self.angle_rot=angle_rot
        self.angle_rot_err=angle_rot_err
        self.energy_frac_FWHM=energy_frac_FWHM
        self.energy_frac_FWHM_err=energy_frac_FWHM_err
        self.microns_per_pixel=microns_per_pixel
        self.microns_per_pixel_err=microns_per_pixel_err

    def get_spot_properties_lst_sqrd_fit(self, im):
        """

        """
        y_max, x_max=im.shape
        x = np.linspace(0, x_max, x_max)#*microns_per_pixel
        y = np.linspace(0, y_max, y_max)#*microns_per_pixel
        X, Y = np.meshgrid(x, y)
        bkg_counts=np.mean(im[0:277, 0:44])
        im=im-bkg_counts

        # estimates for fitted elliptical gaussian properties
        peak_amp=np.amax(im)
        centre_x_pos, sigma_x=calc_moments_spot(im, x, peak_amp, 0)
        centre_y_pos, sigma_y=calc_moments_spot(im, y, peak_amp, 1)
        angle_rot=np.arctan(sigma_x/sigma_y)
        initial_guess = [peak_amp*0.9,centre_x_pos,centre_y_pos,sigma_x,sigma_y,angle_rot,bkg_counts]

        # least squares fit focal spot to elliptical gaussian
        popt, pcov = opt.curve_fit(two_d_gaussian, [X, Y], im.flatten(), p0=initial_guess, bounds=(0, 5000))

        # calculate energy in FWHM
        ellipse=calc_ellipse(X, Y, popt[1], popt[2], popt[3], popt[4], popt[5])
        counts_in_FWHM=np.sum(im[np.where(ellipse<1.0)])
        frac_total_counts_in_FWHM=counts_in_FWHM/np.sum(im)

        #convert spatial focal spot properties to micron units
        popt[1:5]=popt[1:5]*self.microns_per_pixel
        popt=list(popt)
        popt.append(frac_total_counts_in_FWHM)
        return popt

class Laser:
    """

    """
    def __init__(self, wavelength, refractive_index, FWHM_t, FWHM_t_err, f_number, energy, energy_err, throughput, throughput_err, a0=None, a0_err=None, focal_pos_x=None, focal_pos_x_err=None, focal_pos_y=None, focal_pos_y_err=None, focal_pos_z=None, focal_pos_z_err=None, FWHM_x=None, FWHM_x_err=None, FWHM_y=None, FWHM_y_err=None, angle_rot=None, angle_rot_err=None, energy_frac_FWHM=None, energy_frac_FWHM_err=None, microns_per_pixel=None, microns_per_pixel_err=None):
        self.l0=wavelength
        self.n=refractive_index
        self.FWHM_t=FWHM_t
        self.FWHM_t_err=FWHM_t_err
        self.f_number=f_number
        self.energy=energy
        self.energy_err=energy_err
        self.throughput=throughput
        self.throughput_err=throughput_err
        self.a0=a0
        self.a0_err=a0_err
        #focal_spot=FocalSpot(focal_pos_x, focal_pos_x_err, focal_pos_y, focal_pos_y_err, focal_pos_z, focal_pos_z_err, FWHM_x, FWHM_x_err, FWHM_y, FWHM_y_err, angle_rot, angle_rot_err, energy_frac_FWHM, energy_frac_FWHM_err, microns_per_pixel, microns_per_pixel_err)
        self.focal_spot=FocalSpot(focal_pos_x, focal_pos_x_err, focal_pos_y, focal_pos_y_err, focal_pos_z, focal_pos_z_err, FWHM_x, FWHM_x_err, FWHM_y, FWHM_y_err, angle_rot, angle_rot_err, energy_frac_FWHM, energy_frac_FWHM_err, microns_per_pixel, microns_per_pixel_err)

    def calc_waist(self, z, w0, M, z0):
        waist=(w0**2+M**4*(self.l0/(np.pi*w0))**2*(z-z0)**2)**0.5#w0*np.sqrt(1.0+(((z-focal_pos_z)/Zr)*((z-focal_pos_z)/Zr)))
        return waist

    def calc_Raleigh_Range(self, w0):
        return w0*w0*np.pi/self.l0*self.n

    def calc_peak_intensity(self):
        FWHM_x_cm=self.focal_spot.FWHM_x/10**4
        FWHM_y_cm=self.focal_spot.FWHM_y/10**4
        peak_intensity_W_per_cm2=self.energy*self.throughput*(self.focal_spot.energy_frac_FWHM/0.5)*(4.0*np.log(2.0)/np.pi)**1.5/(self.FWHM_t*FWHM_x_cm*FWHM_y_cm)
        peak_intensity_W_per_cm2_percentage_err=((self.energy_err/self.energy)**2+(self.throughput_err/self.throughput)**2+(self.focal_spot.energy_frac_FWHM_err/self.focal_spot.energy_frac_FWHM)**2+(self.FWHM_t_err/self.FWHM_t)**2+(self.focal_spot.FWHM_x_err/self.focal_spot.FWHM_x)**2+(self.focal_spot.FWHM_y_err/self.focal_spot.FWHM_y)**2)**0.5
        return peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_percentage_err*peak_intensity_W_per_cm2

    def calc_a0(self):
        #l0 in microns, peak intensity in Wcm^-2
        peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_err=self.calc_peak_intensity()
        a0=0.855*self.l0*(peak_intensity_W_per_cm2/10**18)**0.5
        return a0, a0*peak_intensity_W_per_cm2_err/peak_intensity_W_per_cm2
