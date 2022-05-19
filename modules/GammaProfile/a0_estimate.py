#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile

import numpy as np
from scipy.constants import pi,c,alpha,m_e
from scipy.ndimage import median_filter,convolve
from skimage.transform import rotate
from PIL import Image, ImageDraw
from glob import glob
import matplotlib.pyplot as plt
from lib.moments_2d import find_width
from lib.contour_ellipse import contour_ellipse

class a0_Estimator:
	"""
	Class for estimating a0
	Contains methods:

	spot_filtering(self,im) : filters an image with a median filter, background 
						subtraction, and optional smoothing

	get_vardiff(self,im) : finds the difference between the spot width horizontally 
							and vertically

	a0_estimate(self,vardiff,gammas) : estimates the a0 given a 
			 difference between the variances and the electron energy
	"""

	# Initialise
	def __init__(self, rad_per_px, wavelength=0.8e-6, FWHM_t=40.0e-15, medfiltwidth=5, smoothwidth=0, bg_path=None,roi=None):
		self.lambda0 = wavelength
		self.tau = FWHM_t
		self.medfiltwidth = medfiltwidth
		self.smoothwidth = smoothwidth
		self.rad_per_px = rad_per_px
		self.bg_path = bg_path
		self.bg = None
		self.mask = None
		if bg_path is not None:
			self.bg = self.average_background(bg_path)
		if roi is not None:
			self.mask = self.make_roi_mask(roi)

	def average_background(self,bg_path,debug=False):		
		bg_files = glob(bg_path+'/*.tif')
		N_files = len(bg_files)
		av_bg = 0
		for i,bg_file in enumerate(bg_files):
			bg_im = plt.imread(bg_file)
			imout = self.spot_filtering(bg_im)
			av_bg += imout/N_files

		if debug:
			plt.imshow(av_bg,cmap='plasma')
			plt.colorbar()
			plt.title(('Background from %i shots in: \n' + bg_path) % N_files)
			plt.tight_layout()
			plt.savefig('Debug/bg.png')
			plt.close()
			
		return av_bg

	def make_roi_mask(self,roi):
		(width,height) = roi[0]
		polygon = roi[1]
		img = Image.new('L', (width, height), 0)
		ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
		mask = np.array(img)
		return mask

	def spot_filtering(self,im):
		"""
		Despeckle an image and remove the background
		"""
		sz = int(2*np.floor(self.medfiltwidth/2)+1) # Make sure sz is odd
		imout = median_filter(im.astype('double'),size=sz,mode='nearest')

		if self.bg is not None:
			imout -= self.bg

		imout -= np.median(imout)

		if self.mask is not None:
			imout *= self.mask

		if self.smoothwidth>0:
			kernel1d = np.kaiser(self.smoothwidth,14)
			x = np.linspace(-0.5,0.5,self.smoothwidth)
			[X,Y] = np.meshgrid(x,x)
			R = np.sqrt(X**2+Y**2)
			kernel = np.interp(R,x,kernel1d,right=0)
			imout = convolve(imout,kernel,mode='nearest')
		return imout
	
	def get_vardiff(self,im):
		"""
		Find the difference in spot width vertically and horizontally
		Use the 2nd moment in the two axes
		"""
		imout = self.spot_filtering(im)
		xw,yw = find_width(imout)
		vardiff = np.abs(xw**2-yw**2)
		return vardiff*self.rad_per_px**2

	def get_vardiff_rot(self,im,level=0.5,debug=False):
		"""
		Find the difference in spot width in two axes of an ellipse
		Use a contour fit to the chosen level
		Also returns the summed spot intensity and the angle of rotation
		"""
		imout = self.spot_filtering(im)
		[major,minor,x0,y0,phi,gof] = contour_ellipse(imout, level)
		imrot = rotate(imout,180.0*phi/np.pi)
		xw,yw = find_width(imrot)
		vardiff = np.abs(xw**2-yw**2)
		
		spot = imout>level*np.max(imout)
		spotMean = np.mean(imout[spot])

		if debug:
			plt.imshow(imrot,cmap='plasma')
			plt.tight_layout()
			N = len(glob('Debug/rot_*.png'))
			plt.savefig('Debug/rot_%i.png' % N)
			plt.close()

		return vardiff*self.rad_per_px**2,spotMean,phi*180/pi,gof

	def get_vardiff_contour(self,im,level=0.5):
		"""
		Find the difference in spot width in two axes of an ellipse
		Use a contour fit to the chosen level
		Also returns the summed spot intensity and the angle of rotation
		"""
		imout = self.spot_filtering(im)
		[major,minor,x0,y0,phi,gof] = contour_ellipse(imout, level)
		vardiff = np.abs(major**2-minor**2) / (-2*np.log(level))
		
		spot = imout>level*np.max(imout)
		spotMean = np.mean(imout[spot])

		return vardiff*self.rad_per_px**2,spotMean,phi*180/pi,gof

	def get_variances_contour(self,im,level=0.5):
		"""
		Find the difference in spot width in two axes of an ellipse
		Use a contour fit to the chosen level
		Also returns the summed spot intensity and the angle of rotation
		"""
		imout = self.spot_filtering(im)
		[major,minor,x0,y0,phi,gof] = contour_ellipse(imout, level)
		var_major = major**2 / (-2*np.log(level))
		var_minor = minor**2 / (-2*np.log(level))
		
		spotMax = np.max(imout)

		return var_major*self.rad_per_px**2,var_minor*self.rad_per_px**2,spotMax,phi*180/pi,gof

	def get_debug_image(self,im,level=0.5):
		"""
		Return the filtered spot only for debugging
		"""
		imout = self.spot_filtering(im)
		return imout

	def get_spot_brightness(self,im):
		"""
		Estimate gamma brightness from the brightest pixel after filtering
		"""
		imout = self.spot_filtering(im)
		im_max = np.max(imout)
		return im_max

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

