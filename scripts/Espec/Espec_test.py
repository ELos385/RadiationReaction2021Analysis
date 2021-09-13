#Espec_test.py

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *

# start with an example shot - same workflow as used in LivePlotting

date = '20210610'
run = 'run18'
shot='Shot020'

espec_diags = ['espec1', 'espec2']
spec_obj = []
file_ext = '.tif'

fig, axes = plt.subplots(2,1)

for n,spec in enumerate(espec_diags):
    run_name = date+'/run99'

    spec_o = Espec(run_name,shot_num=1,img_bkg=None,diag=spec)
    spec_obj.append(spec_o)
    
    filepath = ROOT_DATA_FOLDER + '/' + spec + '/' +  date + '/' + run +'/' + shot + file_ext
    im = spec_o.get_image(filepath)
    x, y = spec_o.xaxis, spec_o.yaxis
    
    axes[n].imshow(im)
    """
    if hasattr(spec_obj[-1], 'p_labels'):
        for p in spec_obj[-1].p_labels:
            win.docks[spec].widgets[0].view.addLine(x=p)
    """

print('finished')