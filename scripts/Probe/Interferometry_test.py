#Espec_test.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Probe.Interferometry import *

# start with an example shot - same workflow as used in LivePlotting

date = '20210621'
run = 'run04'

#date = '20210520'
#run = 'run04'

shot = 7 #shot='Shot002'
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/run99'
LMI = Interferometry(run_name,shot_num=1,diag=diag)


#%%
# grab a shot 
l = [date, run, shot]
filepath = LMI.get_filepath(l)
#filepath = ROOT_DATA_FOLDER + '/' + diag + '/' +  date + '/' + run +'/' + shot + file_ext
im = LMI.get_raw_image(filepath)
extent = LMI.raw_img_extent

plt.figure()
plt.imshow(im, extent=extent)


theta, offset = LMI.get_channel_info(filepath)
phase_o = LMI.get_phase_map(filepath)
t,b,l,r = np.copy(LMI.fringes_roi)

m = np.tan(theta * np.pi/180.0)
x = np.arange(l,r) # get from LMS or LMI as rough collision point
y = (m * (x-l)) + offset + t

e = np.copy(LMI.fringes_img_extent)
plt.figure()
cax = plt.imshow(phase_o.T, extent=e)
plt.title(date + run + 'Shot'+ str(shot))
#plt.plot(x,y,'r--')

plt.colorbar(cax)

#%%
ne = LMI.get_ne(filepath).T
#extent = LMI.raw_img_extent

plt.figure()
plt.imshow(ne)# extent=extent)
plt.show()

"""
n_e_channel = LMI.get_ne_lineout(filepath)
plt.figure()
plt.plot(n_e_channel)
"""
nrows, ncols = ne.shape
midrow = nrows // 2
cw = 7

avg = np.nanmean(ne[midrow - cw: midrow + cw, :], axis=0)
top = np.nanmean(ne[midrow - cw: midrow, :], axis=0)
bottom = np.nanmean(ne[midrow: midrow+cw, :], axis=0)

plt.figure()
plt.plot(avg, label='Average')
plt.plot(top, label='Top')
plt.plot(bottom, label='Bottom')
plt.legend()
plt.grid()
plt.ylabel('$n_e$ [cm$^{-3}$]')
plt.xlabel('pixels')

#%%
plt.show()