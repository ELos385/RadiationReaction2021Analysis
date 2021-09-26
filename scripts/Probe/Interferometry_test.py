#Interferometry_test.py

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

#date = '20210621'
#run = 'run04'

date = '20210622'
run = 'run09'

shot = 143 #shot='Shot002'
file_ext = '.TIFF'

# get the analysis object
diag = 'LMI'
run_name = date+'/' + 'run11'
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
t,b,l,r = LMI.fringes_roi
plt.plot([l, r, r, l, l], [t, t, b, b, t], 'y-')




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
extent = LMI.fringes_img_extent_mm

plt.figure()
cax = plt.imshow(ne, extent=extent)
plt.title(date + run + 'Shot'+ str(shot))
plt.colorbar(cax)
plt.show()

#LMI.channel_width = 5
ne_lineouts = LMI.get_ne_lineout(filepath)
x = LMI.ne_x_mm
plt.figure()

lineObj = plt.plot(x, ne_lineouts.T)
plt.legend(lineObj, ('Average', 'Top', 'Bottom'))
plt.grid()
plt.ylabel('$n_e$ [cm$^{-3}$]')
plt.xlabel('$x$ [mm]')


popt, perr = LMI.get_guass_fit_ne(filepath, plotter=True)


#%%
plt.show()