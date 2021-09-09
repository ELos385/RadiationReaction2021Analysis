#Timing_analysis_test.py

import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Timing.Timing import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# start with an example shot - same workflow as used in LivePlotting

date = '20210618'
run = 'run18'
shot='Shot020'

diag = 'LA3timing'
file_ext = '.TIFF'
filepath = ROOT_DATA_FOLDER + '/' + diag + '/' +  date + '/' + run +'/' + shot + file_ext
drs = date + run + shot

delay = Timing()
im = delay.get_raw_img(filepath)

d = delay.get_delay_from_path(filepath)
str_d = ('%.0f' % d) +' fs'

plt.figure()
plt.imshow(im), plt.title(drs+'\n'+str_d)


#%%
# Grab entire data of diagnostic from given run
diag_timing='LA3timing'

# make sure path to folder for this diag is added to HOME/lib/__init__.py 
# using register_data_loader, otherwise get a KeyError
# single_shot_mode=True for anything that's not a burst run(?)
delay_pipeline = DataPipeline(diag_timing, delay.get_delay_from_img, single_shot_mode=True)

shot_num_delay, delay_data = delay_pipeline.run('%s/%s'%(date, run))
shot_num_delay=np.array(shot_num_delay)

#%% 
# plot delay for the run
plt.figure()
plt.plot(shot_num_delay, delay_data, '.-')
plt.title('%s/%s'%(date, run)), plt.xlabel('Shot #'), plt.ylabel('Delay [fs]')
plt.grid()
