#GammaSpec_fit_Ec.py
"""
GammaStack Image processing and spectrum fitting for Brems data,
handles one shot at a time.
"""

import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from skimage.io import imread

from setup import *
from lib.general_tools import *
from lib.pipeline import *
from modules.GammaSpec.GammaSpecProc import *

diag='CsIStackSide'#'CsIStackTop'#'CsIStackSide'
date='20210608'
run='run10'
shot='Shot029'

# date='20210604'
# run='run20'
# shot='Shot012'

pathT=ROOT_DATA_FOLDER+diag+'/'+date+'/'+run+'/'+shot+'.tif'
GammaStack_Img=imread(pathT)

#load crystal properties from mat file
coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
#load energy deposition info from mat file
Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)

CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, kernel=3, debug=False)

# CsIStack.plot_contours(GammaStack_Img)

guess=[300.0]
measured_signal_summed_over_columns=CsIStack.get_measured_signal_summed_over_columns(GammaStack_Img)#, get_total_counts=False, get_beam_pointing=False)
popt, pcov=CsIStack.least_sqrs_Ec_Brems(guess, measured_signal_summed_over_columns)
print('Ec=%s, sigma Ec=%s'%(popt, pcov))

Gamma_energy_spec=CsIStack.calc_Brems_energy_spec(popt[0])
#
plt.plot(CsIStack.Egamma_MeV_interp, Gamma_energy_spec, label='E$_{crit}=%s$'%(popt[0]))
plt.xlabel('Energy (MeV)')
plt.ylabel('$dN/d\gamma$')
plt.legend()
plt.show()

predicted_signal_summed_over_columns=CsIStack.calc_theoretical_signal_summed_over_columns(None, guess[0])
filter_nos=np.linspace(1, CsIStack.N_crystals_X, CsIStack.N_crystals_X)

plt.plot(filter_nos, predicted_signal_summed_over_columns)
plt.plot(filter_nos, measured_signal_summed_over_columns)
plt.xlabel('filter numbers')
plt.ylabel('normalised counts')
plt.show()
