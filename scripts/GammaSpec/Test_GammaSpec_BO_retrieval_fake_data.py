#Test_GammaSpec_BO_retrieval_fake_data.py

"""
GammaStack Image processing and spectrum fitting for Compton data,
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
# date='20210608'
# run='run10'
# shot='Shot029'

date='20210604'
run='run20'
shot='Shot012'

pathT=ROOT_DATA_FOLDER+diag+'/'+date+'/'+run+'/'+shot+'.tif'
GammaStack_Img=imread(pathT)

#load crystal properties from mat file
coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
#load energy deposition info from mat file
Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)

CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, kernel=3, debug=False)
#CsIStack.plot_contours(GammaStack_Img)

# measured_signal_summed_over_columns=CsIStack.get_measured_signal_summed_over_columns(GammaStack_Img)#, get_total_counts=False, get_beam_pointing=False)

N_spectra=20
Ec=16.0
height=0.05
offset=0.93
spec=CsIStack.calc_Compton_energy_spec(Ec)
fake_signal_summed_over_columns=CsIStack.calc_theoretical_signal_summed_over_columns(None, Ec, height, offset)
inv_CsIEnergy_ProjZ_interp=pinv(CsIStack.CsIEnergy_ProjZ_interp)
# dN_dE_test=np.matmul(measured_signal_summed_over_columns, inv_CsIEnergy_ProjZ_interp)

fake_signal_summed_over_columns_training_set=np.zeros((N_spectra, CsIStack.N_crystals_X))

GammaSpectra=np.zeros((N_spectra, len(CsIStack.Egamma_MeV_interp)))
spectrum=CsIStack.calc_Compton_energy_spec(Ec)
for i in range(N_spectra):
    fake_signal_summed_over_columns_training_set[i, :]=np.random.normal(fake_signal_summed_over_columns, fake_signal_summed_over_columns*0.1, len(fake_signal_summed_over_columns))
    GammaSpectra[i, :]=np.matmul(fake_signal_summed_over_columns_training_set[i, :], inv_CsIEnergy_ProjZ_interp)
    plt.plot(CsIStack.Egamma_MeV_interp, GammaSpectra[i, :])
# plt.show()

# Bayesian Optimisation routine
# spec_b_opt, err_spec_b_opt=CsIStack.bayesian_opt_spec(fake_signal_summed_over_columns_training_set)
# plt.plot(CsIStack.Egamma_MeV_interp, spec_b_opt, color='r')
plt.plot(CsIStack.Egamma_MeV_interp, spec, color='b')
#plt.fill_between(CsIStack.Egamma_MeV_interp, spec_b_opt-err_spec_b_opt, spec_b_opt+err_spec_b_opt, color='r')
plt.show()
