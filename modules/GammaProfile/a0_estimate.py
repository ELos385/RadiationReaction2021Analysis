#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile

import numpy as np
from scipy.constants import pi,c,alpha,m_e
from scipy.signal import medfilt
from lib.moments_2d import find_width
from lib.contour_ellipse import contour_ellipse
	
def spot_filtering(im,medfiltwidth=5,threshold=1):
	"""
	Despeckle an image and remove a constant background
	"""
	sz = int(2*np.floor(medfiltwidth/2)+1) # Make sure sz is odd
	despeckled = medfilt(im,kernel_size=sz)
	bg = threshold*np.median(despeckled)
	imout = despeckled - bg

	return imout	

def get_vardiff(im,rad_per_px):
	"""
	Find the difference in spot width vertically and horizontally
	"""
	xw,yw = find_width(im)
	vardiff = np.abs(xw**2-yw**2)*rad_per_px**2

	return vardiff

def get_vardiff_contour(im,rad_per_px,contour_level=0.5):
	"""
	Find the difference in spot width vertically and horizontally
	Use a contour fit to the chosen level
	"""
	[major,minor,x0,y0,phi] = contour_ellipse(im,contour_level,debug=True)
	vardiff = np.abs(major**2-minor**2)*rad_per_px**2

	return vardiff

class a0_Estimator:
	"""
	Class for estimating a0
	Contains methods:

	get_vardiff(self,im) : reads an image and finds the difference
			 between the spot width horizontally and
			 vertically

	a0_estimate(self,vardiff,gammas) : estimates the a0 given a 
			 difference between the variances and the electron energy
	"""
	# Initialise
	def __init__(self, rad_per_px, wavelength=0.8e-6, FWHM_t=40.0e-15, medfiltwidth=5, threshold=1):
		self.lambda0 = wavelength
		self.tau = FWHM_t
		self.medfiltwidth = medfiltwidth
		self.threshold = threshold
		self.rad_per_px = rad_per_px
	
	def get_vardiff(self,im):
		imout = spot_filtering(im, medfiltwidth=self.medfiltwidth, threshold=self.threshold)
		vardiff = get_vardiff(imout,self.rad_per_px)
		return vardiff

	def get_vardiff_contour(self,im):
		imout = spot_filtering(im, medfiltwidth=self.medfiltwidth, threshold=self.threshold)
		vardiff = get_vardiff_contour(imout,self.rad_per_px)
		return vardiff

	def a0_estimate_av(self,vardiff,gammai,gammaf):
		"""
		Estimate a0 in a way which is model independent
		"""
		a0 = np.sqrt(4*np.sqrt(2))*np.sqrt(gammaf*gammai*vardiff)
		return a0

	def a0_estimate_cl(self,vardiff,gammai,tau=40.0e-15, lambda0=0.8e-6):
		"""
		Estimate a0, assuming a Gaussian beam and classical RR
		"""
		R = (8.0/5.0)*gammai**2 * vardiff - 1.0
		omega0 = 2*pi*c/lambda0
		g2 = omega0*tau*np.sqrt(pi/(4*np.log(2)))
		a0 = np.sqrt((3*m_e*R)/(2*alpha*gammai*omega0*g2))
		return a0

