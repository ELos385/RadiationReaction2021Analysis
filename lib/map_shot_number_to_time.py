import numpy as np
import matplotlib.pyplot as plt;
import os, sys;
sys.path.append('../../')
sys.path.append('../')
from lib.general_tools import *


date='20210604'#'20210604'
in_filepath='/Volumes/Seagate Expansion Drive/Radiation_Reaction_2021/Analysis/GammaSpec/Bayesian Inference/%s/'%(date)
files=os.listdir(in_filepath)
print(files)
GSN_dict={}
GSN_date_dict={
'20210604':[283366, 284708],
'20210620':[293611, 295698]}

GSN_arr=np.arange(GSN_date_dict[date][0], GSN_date_dict[date][1])

file_path_GSN='/Volumes/Seagate Expansion Drive/Radiation_Reaction_2021/GSNs_ECAT/%s.csv'%(date)
data=pd.read_csv(file_path_GSN)
Ids=np.array(data['Id'])[::-1]
times=np.array(data['Time'])[::-1]

print(Ids[0])
print(Ids.shape)
print(times.shape)
print(GSN_arr.shape)

times=times[(Ids>=GSN_date_dict[date][0])&(Ids<=GSN_date_dict[date][1])]
Ids=Ids[(Ids>=GSN_date_dict[date][0])&(Ids<=GSN_date_dict[date][1])]

shots_before_run=0
for i in range(0, len(files)):
    run=files[i][11:-4]
    full_path=in_filepath+files[i]
    gamma_spec_data=load_object(full_path)[run]
    kes=np.asarray([list(gamma_spec_data.keys())]).flatten()
    run_dict={}
    run_dict['Shot numbers']=kes
    run_dict['GSN']=GSN_arr[shots_before_run:shots_before_run+len(kes)]
    run_dict['Time']=times[shots_before_run:shots_before_run+len(kes)]
    print(len(kes))
    print(len(times[shots_before_run:shots_before_run+len(kes)]))
    print(len(GSN_arr[shots_before_run:shots_before_run+len(kes)]))
    GSN_dict[run]=run_dict
    shots_before_run+=len(kes)

    print(run_dict['Shot numbers'][0])
    print(run_dict['Shot numbers'][-1])
    print(run_dict['GSN'][0])
    print(run_dict['GSN'][-1])

out_filepath='/Users/ee.los/Documents/GitHub/RadiationReaction2021Analysis/lib/shot_2_GSN/'
if not os.path.exists(out_filepath):
    os.makedirs(out_filepath)
filename=out_filepath+'%s.pkl'%(date)
save_object(GSN_dict, filename)


    #GSN_dict[]
#gamma_spec_data=load_object(in_filepath)
