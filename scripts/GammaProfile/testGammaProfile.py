#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile screen

from lib.pipeline import DataPipeline
from modules.GammaProfile.a0_estimate import a0_Estimator
from calib.GammaProfile import rad_per_px

date= '20210620'
run= 'run09'
diag='GammaProfile'

a0_Est = a0_Estimator(rad_per_px=rad_per_px)
a0_pipeline = DataPipeline(diag,a0_Est.get_a0, single_shot_mode=True)
shot_num, a0_data = a0_pipeline.run('%s/%s'%(date, run))
shot_num = np.array(shot_num)
