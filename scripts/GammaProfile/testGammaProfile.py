#!/usr/bin/python3
# Author: Chris Arran
# Date: September 2021
#
# Aims to estimate the a0 from the gamma profile screen

import numpy as np
from lib.pipeline import DataPipeline
from modules.GammaProfile.a0_estimate import a0_Estimator, a0_estimate_av
from calib.GammaProfile import rad_per_px

date= '20210620'
run= 'run09subset'
diag='GammaProfile'

gammai = 800/0.511
gammaf = 600/0.511

a0_Est = a0_Estimator(rad_per_px=rad_per_px)
a0_pipeline = DataPipeline(diag,a0_Est.get_vardiff, single_shot_mode=True)
shot_num, vardiff_data = a0_pipeline.run('%s/%s'%(date, run))
a0_data = a0_estimate_av(vardiff_data,gammai,gammaf)
