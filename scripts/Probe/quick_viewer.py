import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.sql_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

#date = '20210621'
#run = 'run04'

date = '20210520'
run = 'run17'

shot = 1
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/' + run
LMI = Interferometry(run_name,shot_num=1,diag=diag)

#%%

for shot in range(1,36+1):
    l = [date, run, shot]
    filepath = LMI.get_filepath(l)    
    img =  LMI.get_raw_image(filepath)
    img = LMI.get_phase_map(filepath).T
    plt.figure()
    plt.imshow(img)
    plt.title(shot)
    