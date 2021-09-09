import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sqlite3
from glob import glob
from datetime import datetime
import sys
sys.path.append('../../')
from setup import *

from pathlib import Path
DATA_PATH = Path(ROOT_DATA_FOLDER)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4) # changed pickle protocol to 4 for compatibility with mirage, python 3.7

def load_object(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)

def imagesc(I,ax = None,  x=None, y=None, **kwargs):
    """ display image with axes using pyplot
    recreates matlab imagesc functionality (roughly)
    argument I = 2D numpy array for plotting
    kwargs:
        ax = axes to plot on, if None will make a new one
        x = horizontal  axis - just uses first an last values to set extent
        y = vetical axis like x
        **kwargs anthing else which is passed to imshow except extent and aspec which are set
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    if x is None:
        Nx = np.size(I, axis=1)
        x = np.arange(Nx)
    if y is None:
        Ny = np.size(I, axis=0)
        y = np.arange(Ny)
    ext = (x[0], x[-1], y[-1], y[0])
    return ax.imshow(I, extent=ext, aspect='auto', **kwargs)
### functions for determining which calibration file to use

def get_eSpec_Calib_paths(diag, date):
    ''' Returns paths of energy and charge calibration files given the
    date and name of the diagnostic.
    '''
    dates_in_path_dir_transform = []
    dates_in_path_dir_disp_cal = []

    base_path = CAL_DATA+'/%s/'%(diag)
    for name in os.listdir(base_path):
        print(name)
        prefix_t=name[0:17]
        prefix_d=name[0:16]
        if prefix_t=='espec2_transform_' or prefix_t=='espec1_transform_':
            dates_in_path_dir_transform.append(int(name[17:25]))
        elif prefix_d=='espec2_disp_cal_' or prefix_d=='espec1_disp_cal_':
            dates_in_path_dir_disp_cal.append(int(name[16:24]))
    dates_in_path_dir_transform=np.array(dates_in_path_dir_transform).astype(int)
    dates_in_path_dir_disp_cal=np.array(dates_in_path_dir_disp_cal).astype(int)
    pkl_date=sorted(dates_in_path_dir_transform, key=lambda i: abs(i - int(date)))[0]
    tForm_filepath = CAL_DATA+'/%s/%s_transform_%s_run01_shot001.pkl'%(diag, diag, pkl_date)
    Espec_cal_filepath = CAL_DATA+'/%s/%s_disp_cal_%s_run01_shot001.mat'%(diag, diag, dates_in_path_dir_disp_cal[0])
    return tForm_filepath, Espec_cal_filepath

def choose_cal_file(run_name,shot,diag,file_pref,cal_data_path=None):
    run_dt, run_num = get_run_name_info(run_name)
    c_path_sel = None

    cal_paths = get_cal_files(diag=diag,file_pref=file_pref,cal_data_path=cal_data_path)
    N_t_path = len(cal_paths)

    if N_t_path==0:
        print("No %s files found at %s" % (file_pref, cal_data_path))
        return c_path_sel

    dts_checked = []
    for c_path in cal_paths:
        # print('c_path: ', c_path)
        c_path_dt, c_path_run_num, c_path_shot_num =get_cal_path_info(c_path,file_pref=file_pref)
        pre_data = is_arg1_geq_arg2((run_dt, run_num, shot ),
                                    (c_path_dt, c_path_run_num, c_path_shot_num ))
        dts_checked.append(pre_data)
        if pre_data:
            if c_path_sel is None:
                c_path_sel = c_path
                c_path_sel_info = (c_path_dt,c_path_run_num,c_path_shot_num)

            else:
                post_other =is_arg1_geq_arg2((c_path_dt, c_path_run_num, c_path_shot_num ),
                                    c_path_sel_info)
                if post_other:
                    c_path_sel = c_path
                    c_path_sel_dt = c_path_dt
                    c_path_sel_info = (c_path_dt,c_path_run_num,c_path_shot_num)

    if c_path_sel is None and not any(dts_checked):
        print('No %s files found in %s to exist from before current shot.' % (file_pref, cal_data_path))
        print('The current shot is %s Shot %s' % (run_name, shot))

    return c_path_sel

def get_cal_files(diag,file_pref,cal_data_path=None):
    if cal_data_path is None:
        cal_data_path = CAL_DATA

    cal_paths =  glob(os.path.join(cal_data_path,diag,file_pref+'*'))

    return cal_paths

def get_cal_path_info(filepath,file_pref):
    cal_path_date, cal_path_run_shot = filepath.split(file_pref+'_')[1].split('_run')
    cal_path_run_str, cal_path_shot_str = cal_path_run_shot.split('.')[0].split('_shot')
    cal_path_dt =  datetime.strptime(cal_path_date, '%Y%m%d')
    cal_path_run_num = int(cal_path_run_str)
    cal_path_shot_num = int(cal_path_shot_str)
    return cal_path_dt, cal_path_run_num, cal_path_shot_num

def get_run_name_info(run_name):
    run_date, run_num = run_name.split('/')
    run_dt = datetime.strptime(run_date, '%Y%m%d')
    run_num = int(run_num.split('run')[1])
    return run_dt, run_num

def is_arg1_geq_arg2(dt_shot_run_tup1,dt_shot_run_tup2):

    if (dt_shot_run_tup1[0]-dt_shot_run_tup2[0]).total_seconds()==0:
        # same day
        if dt_shot_run_tup1[1]==dt_shot_run_tup2[1]:
            #same run
            if dt_shot_run_tup1[2]>=dt_shot_run_tup2[2]:
                answer = True
            else:
                answer = False
        elif dt_shot_run_tup1[1]>dt_shot_run_tup2[1]:
            answer = True
        else:
            answer = False
    elif (dt_shot_run_tup1[0]-dt_shot_run_tup2[0]).total_seconds()>0:
        answer = True
    else:
        answer = False

    return answer

def calc_COW(img,X=None,Y=None,img_thresh=0.5):
    iSel = img>img_thresh
    if (X is None) or (Y is None):
        Ny,Nx = np.shape(img)
        X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    c_x = np.sum(X[iSel]*img[iSel])/np.sum(img[iSel])
    c_y = np.sum(Y[iSel]*img[iSel])/np.sum(img[iSel])
    return c_x,c_y

def glob_path(p):
    return glob(str(p))

def d(x):
    return np.abs(np.mean(np.diff(x)))


def smooth_gauss(x,y,sigma_x):
    X1,X2 = np.meshgrid(x,x)
    W = np.exp(-(X1-X2)**2/(2*sigma_x**2))
    y_smooth = np.sum(W*y,axis=1)/np.sum(W,axis=1)
    return y_smooth


def get_file_path(diag,run_name,burst_number,shot_number=None,verbose=False):
    if shot_number is not None:
        shot_str = f'Shot{shot_number:03}.*'
    else:
        shot_str = 'Shot*.*'
    if burst_number is None:
        run_stem = DATA_PATH / diag / run_name / shot_str
    else:
        run_stem = DATA_PATH / diag / run_name / f'Burst{burst_number:03}' / shot_str
    if verbose:
        print(run_stem)


    file_list = glob_path(run_stem)
    if len(file_list)==1:
        file_path = file_list[0]
    elif len(file_list)==0:
        file_path = None
        if verbose:
            print('No file found')
    else:
        file_path = sorted(file_list)
        if verbose:
            print('Multiple files found')
    return file_path

def normalise(I):
    return I/np.max(np.abs(I))



class  Read_SQL_shot_summary:
    def __init__(self):
        #self.sql_path = DATA_PATH / 'data.sqlite'
        self.sql_path = str(DATA_PATH) + '/' +  'data.sqlite'

    def get_all(self):
        with sqlite3.connect(self.sql_path) as conn:
            result =  pd.read_sql_query(f"select * from shot_summary",conn)
        return result

    def get_run(self,run_name, burst=None):
        with sqlite3.connect(self.sql_path) as conn:
            df = pd.read_sql_query(f"select * from shot_summary where run = '{run_name}'",conn)
        df = df.sort_values('timestamp')

        if burst is not None:
            df = df[df['shot_or_burst']==burst]
        df = df.reset_index(drop=True)
        return df

    def get_run_names(self):
        with sqlite3.connect(self.sql_path) as conn:
            df = pd.read_sql_query("select * from shot_summary",conn)
        _,unique_inds = np.unique(df['run'].values,return_index=True)
        return df['run'].values[np.sort(unique_inds)]
