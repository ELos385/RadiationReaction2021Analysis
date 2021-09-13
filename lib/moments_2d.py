#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to calculate nth moments of a 2D scalar field - this can be used for instance to estimate the center of a spot and its widths
# Capital letters are generally matrices, lower case are vectors or scalars

import numpy as np

# Return vectors containing the nth moments taken along each axis
def moments_2d(Arr,n):
	x = np.arange(Arr.shape[0])
	y = np.arange(Arr.shape[1])
	X,Y = np.meshgrid(x,y,indexing='ij')
	
	xmoment = np.sum(Arr * X**n, axis=0)
	ymoment = np.sum(Arr * Y**n, axis=1)
	
	return xmoment,ymoment
	
# Estimate the weighted mean center of a spot using the 1st moment
def find_center(Arr):
	x0,y0 = moments_2d(Arr,0)
	x1,y1 = moments_2d(Arr,1)
	
	xc = np.sum(x1)/np.sum(x0)
	yc = np.sum(y1)/np.sum(y0)
	
	return np.array([xc,yc])
	
# Estimate the spot width using the 2nd moment
def find_width(Arr):
	x0,y0 = moments_2d(Arr,0)
	x1,y1 = moments_2d(Arr,1)
	x2,y2 = moments_2d(Arr,2)
	
	xvar = (x2/x0)-(x1/x0)**2
	yvar = (y2/y0)-(y1/y0)**2
	xw = np.sqrt(np.nansum(x0*xvar)/np.sum(x0))
	yw = np.sqrt(np.nansum(y0*yvar)/np.sum(y0))
	
	return xw,yw
	

