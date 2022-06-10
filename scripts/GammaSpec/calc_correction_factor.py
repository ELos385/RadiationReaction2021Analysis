#calc_correction_factor.py

import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv, kn, expi
from scipy.constants import e, c, epsilon_0, m_e, Boltzmann, hbar, Planck
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

dict_corr_factor={}
diag_arr=np.array(['CsIStackTop', 'CsIStackSide'])
date='20210622'
run='run19'
# left out shots 1 and 16 as they were anomalous
shot_no_arr=np.array(['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20'])

for k in range(0, len(diag_arr)):
    diag=diag_arr[k]

    # get reference image
    path_calib=ROOT_DATA_FOLDER+diag+'/'+date+'/'+bkg_img_dict[date][0]+'/'+"Shot00%s.tif"%(bkg_img_dict[date][1])
    calib_img=imread(path_calib)

    Geant_signal_rows_arr=[]
    measured_signal_rows_arr=[]
    correction_factor_arr=[]

    for i in range(0, len(shot_no_arr)):
        shot_no=shot_no_arr[i]
        pathT=ROOT_DATA_FOLDER+diag+'/'+date+'/'+run+'/Shot0%s.tif'%(shot_no)
        GammaStack_Img=imread(pathT)

        #load crystal properties from mat file
        coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
        #load energy deposition info from mat file
        Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)

        CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, calib_img, kernel=3, debug=False)
        #CsIStack.plot_contours(GammaStack_Img)

        #measured_signal_summed_over_columns=CsIStack.get_measured_signal_summed_over_columns(GammaStack_Img)#, get_total_counts=False, get_beam_pointing=False)
        measured_signal_rows_and_cols=CsIStack.get_measured_signal_rows_and_cols(GammaStack_Img)#, get_total_counts=False, get_beam_pointing=False)
        measured_signal_rows=np.sum(measured_signal_rows_and_cols, axis=0)/max(np.sum(measured_signal_rows_and_cols, axis=0))
        measured_signal_rows_arr.append(measured_signal_rows)


        GEANT_sims_filename=CALIB_FOLDER+"/GammaStack/"+date+"/"+run+"/ArbBrems_Shot0%s_spectra.mat"%(shot_no)
        GEANT_data=loadmat(GEANT_sims_filename)
        if (diag=='CsIStackTop'):
            CsI_top_GEANT=GEANT_data["CsIEnergy_VDual_2"]
            GEANT_signal_rows_and_cols=CsI_top_GEANT[0:measured_signal_rows_and_cols.shape[1], 0:measured_signal_rows_and_cols.shape[0]]
        if (diag=='CsIStackSide'):
            CsI_side_GEANT=GEANT_data["CsIEnergy_HDual_2"]
            GEANT_signal_rows_and_cols=CsI_side_GEANT[0:measured_signal_rows_and_cols.shape[1], 0:measured_signal_rows_and_cols.shape[0]]

        GEANT_signal_rows_and_cols=GEANT_signal_rows_and_cols#/np.amax(GEANT_signal_rows_and_cols)
        GEANT_signal_rows=np.sum(GEANT_signal_rows_and_cols, axis=1)/max(np.sum(GEANT_signal_rows_and_cols, axis=1))
        Geant_signal_rows_arr.append(GEANT_signal_rows)

        correction_factor_1d=GEANT_signal_rows/measured_signal_rows
        correction_factor_arr.append(correction_factor_1d)

    filter_nos_X=np.linspace(1, CsIStack.N_crystals_X, CsIStack.N_crystals_X)

    correction_factor_arr=np.asarray(correction_factor_arr)
    Geant_signal_rows_arr=np.asarray(Geant_signal_rows_arr)
    measured_signal_rows_arr=np.asarray(measured_signal_rows_arr)

    mean_Geant_rows, std_Geant_rows=np.mean(Geant_signal_rows_arr, axis=0), np.std(Geant_signal_rows_arr, axis=0)
    mean_measured_rows, std_measured_rows=np.mean(measured_signal_rows_arr, axis=0), np.std(measured_signal_rows_arr, axis=0)
    mean_correction_factor, std_correction_factor=np.mean(correction_factor_arr, axis=0), np.std(correction_factor_arr, axis=0)

    for i in range(len(correction_factor_arr)):
        plt.plot(filter_nos_X, Geant_signal_rows_arr[i], color='r', label='Geant')
        plt.plot(filter_nos_X, measured_signal_rows_arr[i], color='b', label='measured')
    plt.show()

    plt.plot(filter_nos_X, mean_Geant_rows, color='r', label='Geant')
    plt.fill_between(filter_nos_X, mean_Geant_rows-std_Geant_rows, mean_Geant_rows+std_Geant_rows, color='r', alpha=0.5)
    plt.plot(filter_nos_X, mean_measured_rows, color='b', label='Measured')
    plt.fill_between(filter_nos_X, mean_measured_rows-std_measured_rows, mean_measured_rows+std_measured_rows, color='b', alpha=0.5)
    plt.plot(filter_nos_X, mean_correction_factor, color='g', label='Correction factor')
    plt.fill_between(filter_nos_X, mean_correction_factor-std_correction_factor, mean_correction_factor+std_correction_factor, color='g', alpha=0.5)
    plt.xlabel('Crystal Number')
    plt.ylabel('Normalised Counts')
    plt.tight_layout()
    plt.legend()
    plt.show()

    dict_corr_factor[diag]=mean_correction_factor
    print("dict_corr_factor[diag]  = %s"%dict_corr_factor[diag])

outfile=CALIB_FOLDER+"/GammaStack/gamma_stack_correction_factor.pkl"
with open(outfile, 'wb') as handle:
    pickle.dump(dict_corr_factor, handle, protocol=pickle.HIGHEST_PROTOCOL)
