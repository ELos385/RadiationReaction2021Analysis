#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile screen
# Added the espec to pull out gamma_i and gamma_f

import numpy as np
from config import HOME
from lib.pipeline import DataPipeline
from modules.Espec.espec_processing import Espec_proc
from modules.GammaProfile.a0_estimate import a0_Estimator, a0_estimate_av
from calib.GammaProfile import rad_per_px

date= '20210620'
run= 'run09'

tForm_filepath = HOME + 'calib/espec1/espec1_transform_20210621_run01_shot001.pkl'
Espec_cal_filepath = HOME + 'calib/espec1/espec1_disp_cal_20210527_run01_shot001.mat'

gamma_profile='GammaProfile'
espec = 'espec1'

espec_Proc = Espec_proc(tForm_filepath,Espec_cal_filepath)
espec_pipeline = DataPipeline(espec, espec_Proc.mean_and_std_beam_energy, single_shot_mode=True)
shot_num, espec_data = espec_pipeline.run('%s/%s'%(date, run))

gammaf = espec_data[:,0]/0.511	# 90th percentile energies
gammai = np.median(gammaf)	# Assume most shots are misses

a0_Est = a0_Estimator(rad_per_px)
a0_pipeline = DataPipeline(gamma_profile,a0_Est.get_vardiff, single_shot_mode=True)
shot_num, vardiff_data = a0_pipeline.run('%s/%s'%(date, run))

a0_data = a0_estimate_av(vardiff_data,gammai,gammaf)

