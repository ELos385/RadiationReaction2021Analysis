"""Test the corner plot script for analysing correlations 
(originally of optimisation)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from setup import *
from lib.pipeline import *
from lib.general_tools import *
from lib.folder_tools import *
from modules.Espec.Espec import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.data_visualisation import correlationPlot_mod



import pandas as pd

run_name = '20210527/run12'
sScale = [500,25000,400000,0.1]


# copy from Jupyter Notebooks
data_file = '/Volumes/Backup2/2021_Radiation_Reaction/Automation/Outputs/LWFA_GP_20210527_run12.txt'
opt_df = pd.read_csv(data_file,index_col=False,delim_whitespace=True)
data_file = '/Volumes/Backup2/2021_Radiation_Reaction/Automation/Outputs/LWFA_GP__20210527_run12_model.txt'
model_df = pd.read_csv(data_file,index_col=False,delim_whitespace=True)



parameters = opt_df.keys()[2:-2]
opt_flag = []
for key in parameters:
    x = opt_df[key].values
    if len(np.unique(x))>1:
        opt_flag.append(1)
    else:
        opt_flag.append(0)
        


param_list = ['o2','o3','o4','focus','astig0','astig45','coma0','coma90','spherical','target_z']



pLabels = [r'$\beta^{(2)}$' + '\n' + r'[$10^3$ fs$^2$]', r'$\beta^{(3)}$' + '\n' + r'[$10^4$ fs$^3$]',
           r'$\beta^{(4)}$' + '\n' + r'[$10^5$ fs$^4$]', r'$f$' + '\n' + r'[mm]', r'astig0',r'astig45',
           r'coma45',r'coma90',r'spherical',
           r'$Z$' + '\n' + r'[mm]']


pScale = [1e3,1e4,1e5,1,1,1,1,1,1,1]

sScale = [sScale[n] for n in range(len(sScale)) if opt_flag[n]>0]
parameters = [param_list[n] for n in range(len(param_list)) if opt_flag[n]>0]
pLabels = [pLabels[n] for n in range(len(pLabels)) if opt_flag[n]>0]
pScale = [pScale[n] for n in range(len(pScale)) if opt_flag[n]>0]

p0 = opt_df[parameters].values[0]

fig = plt.figure(figsize=(8,5),dpi=150)


fig,axs = correlationPlot_mod(opt_df,parameters,pScale,fig=fig, cmap='viridis',
                              p0 = None, pLabels=None,xlims=None,ylims=None)
fig.subplots_adjust(left=0.15, bottom=0.2, right=0.6, top=0.95, wspace=0.025, hspace=0.025)
