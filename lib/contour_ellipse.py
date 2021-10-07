#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the width of an elliptical spot in two directions

import numpy as np
from skimage.measure import find_contours
from lib.fit_ellipse import fit_ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from glob import glob

def contour_gof(contours,ellipse):
	"""
	Estimate the goodness of fit from the rms distance of contour points 
	from the fitted ellipse
	"""

	[x,y] = contours
	D = np.vstack((x * x, x * y, y * y, x, y, np.ones_like(x)))

	[major,minor,x0,y0,phi] = ellipse
	a = np.zeros(6)
	a[0] = (major*np.sin(phi))**2 + (minor*np.cos(phi))**2
	a[1] = - 2*(major**2 - minor**2)*np.sin(phi)*np.cos(phi)
	a[2] = (major*np.cos(phi))**2 + (minor*np.sin(phi))**2
	a[3] = -2*a[0]*x0 - a[1]*y0
	a[4] = -a[1]*x0 - 2*a[2]*y0
	a[5] = a[0]*x0**2+a[1]*x0*y0+a[2]*y0**2-(major*minor)**2

	residual = np.linalg.multi_dot((a,D,D.T,a))
	return np.sqrt(residual/(len(x)-6))

def plot_contour_ellipse(im,contours,ellipse,path=None):
	"""
	Plots the contour and the fitted ellipse on top of the image im
	contours is an array [x,y] of the contour coordinates
	ellipse is an array [major,minor,x0,y0,phi] of ellipse parameters
	"""

	files = glob(path+"contourFit*.png")
	fname = path+"contourFit%i.png" % len(files)
	fig, ax = plt.subplots(num=len(files))

	ax.imshow(im,cmap='plasma',vmin=0,vmax=np.max(im))

	[x,y] = contours
	ax.plot(x,y,'.',zorder=50)

	[major,minor,x0,y0,phi] = ellipse
	ellipse_obj = Ellipse((x0, y0), major*2, minor*2, (180/np.pi)*phi, 
		            facecolor='none',edgecolor='green',linestyle='-', 
		            zorder=100)
	ax.add_patch(ellipse_obj)
	
	fig.savefig(fname)
	plt.close(fig)

	return fname

def contour_ellipse(im,level=0.5,debug=False,debugpath=None):
	"""
	Finds a contour at the given level and fits it to an ellipse.
	level is given relative to the maximum
	"""
	contours = find_contours(im, level*np.max(im), fully_connected='high')
	flattened = np.concatenate(contours)
	x,y = flattened[:,1],flattened[:,0]

	[major,minor,x0,y0,phi] = fit_ellipse(x,y)
	gof = contour_gof([x,y],[major,minor,x0,y0,phi])
	
	if debug:
		fname = plot_contour_ellipse(im,[x,y],[major,minor,x0,y0,phi],debugpath)
		print("Saved image: " + fname)
		print("Goodness of fit: %.2e" % gof)
		print("Ellipse: %.2f, %.2f, %.2f, %.2f, %.2f" % (major,minor,x0,y0,phi))

	return [major,minor,x0,y0,phi,gof]
	
	
