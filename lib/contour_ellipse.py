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

def plot_contour_ellipse(im,contours,ellipse):
	"""
	Plots the contour and the fitted ellipse on top of the image
	"""

	fig, ax = plt.subplots()

	plt.imshow(im/np.max(im),cmap='plasma',vmin=0)
	[x,y] = contours
	plt.plot(x,y,'.',zorder=50)

	print(ellipse)
	[major,minor,x0,y0,phi] = ellipse
	ellipse_obj = Ellipse((x0, y0), major*2, minor*2, (180/np.pi)*phi, 
		            facecolor='none',edgecolor='green',linestyle='-', 
		            zorder=100)
	ax.add_patch(ellipse_obj)
	
	files = glob("contourFit*.png")
	fname = "contourFit%i.png" % len(files)
	plt.savefig(fname)
	plt.close()

	return fname

def contour_ellipse(im,level=0.5,debug=False):
	"""
	Finds a contour at the given level and fits it to an ellipse.
	level is given relative to the maximum
	"""
	contours = find_contours(im, level*np.max(im), fully_connected='high')
	flattened = np.concatenate(contours)
	x,y = flattened[:,1],flattened[:,0]
	[major,minor,x0,y0,phi] = fit_ellipse(x,y)
	
	if debug:
	    fname = plot_contour_ellipse(im,[x,y],[major,minor,x0,y0,phi])

	return [major,minor,x0,y0,phi]
	
	
