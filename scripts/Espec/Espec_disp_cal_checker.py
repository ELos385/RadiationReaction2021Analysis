#Espec_disp_cal_checker.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *

# grab a disp_cal file
diag = 'espec1'
date = '20210527'
run = 'run01'
shot = 'shot001'
filedir = HOME + '/calib/' + diag + '/'

filename = '%s_disp_cal_%s_%s_%s.mat' % (diag, date, run,  shot)
filepath = filedir + filename

dc = loadmat(filepath)

#%%
# check it

plt.figure()
plt.plot(dc['spec_x_mm'].flatten(), dc['spec_MeV'].flatten())
plt.title('%s %s %s' % (date,run,shot)), plt.xlabel('x [mm]'), plt.ylabel('T [MeV]')
plt.show()