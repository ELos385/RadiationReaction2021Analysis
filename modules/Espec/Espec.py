from config import *
import numpy as np

from modules.Espec import espec_processing
from lib.general_tools import *
import pickle
from scipy.interpolate import interp1d
from skimage.io import imread
from scipy.io import loadmat

class Espec:
    def __init__(self,run_name,shot_num=1,img_bkg=None,diag='espec1',cal_data_path=HOME + '/calib/'):
        self.run_name = run_name
        self.shot_num = shot_num
        self.img_bkg = img_bkg
        self.diag = diag
        self.cal_data_path = cal_data_path

        self.setup_proc(run_name=run_name,shot_num=shot_num,img_bkg=img_bkg)


    def setup_proc(self,run_name=None,shot_num=None,img_bkg=None):
        if run_name is None:
            run_name = self.run_name
        if shot_num is None:
            shot_num = self.shot_num
        if img_bkg is None:
            img_bkg = self.img_bkg

        # paths for image warp and dispersion calibrations files
        try:
            tForm_filepath = choose_cal_file(run_name,shot_num, self.diag,
                            self.diag + '_transform', cal_data_path=self.cal_data_path)
            self.tForm_filepath = tForm_filepath
            
        except IndexError as e:
            print('No transform files found for this run %s' % (run_name))
            print(e)
            return None


        try:
            Espec_cal_filepath = choose_cal_file(run_name,shot_num,self.diag,
                        self.diag + '_disp_cal', cal_data_path=self.cal_data_path)
            self.Espec_cal_filepath = Espec_cal_filepath

        except IndexError as e:
            print('No calibration files found for this run %s' % (run_name))
            print(e)
            return None


        # defaults
        self.xaxis = []
        self.yaxis = []

        if tForm_filepath is None:
            # don't continue with processing
            return None

        if Espec_cal_filepath is None:
            # don't continue with processing = necessary?
            return None

        try:
            tForm = pickle.load(open(tForm_filepath,'rb'))
            print('tForm file loaded was %s' % (tForm_filepath))
        except ValueError as ve:
            print("Pickle can't load ", tForm_filepath)
            print(ve)
            return None

        try:
            Espec_cal = loadmat(Espec_cal_filepath)
            print('disp_cal file loaded was %s' % (Espec_cal_filepath))
        except ValueError as ve:
            print("Can't load ", Espec_cal_filepath)
            print(ve)
            return None   
        
        # setup espec processor
        eSpec_proc = espec_processing.Espec_proc(tForm_filepath,Espec_cal_filepath,
                             img_bkg=img_bkg,use_median=True,kernel_size=None )
        self.eSpec_proc = eSpec_proc

        # image axes
        self.x_mm = eSpec_proc.screen_x_mm
        self.dx = np.mean(np.diff(self.x_mm))
        self.y_mm = eSpec_proc.screen_y_mm
        self.dy = np.mean(np.diff(self.y_mm))

        # energy markers
        spec_x_mm = eSpec_proc.spec_x_mm.flatten()
        spec_MeV = eSpec_proc.spec_MeV.flatten()
        
        # xp = x_MeV_cal(Ep)
        x_MeV_cal = interp1d(spec_MeV, spec_x_mm, kind='linear', copy=True, bounds_error=False, fill_value=np.nan)
        E_labels = np.arange(400,1501,100)
        x_labels= x_MeV_cal(E_labels)
        x2p_func = interp1d(self.x_mm, np.arange(0,len(self.x_mm)), kind='linear', copy=True, bounds_error=False, fill_value=np.nan)

        p_labels = x2p_func(x_labels)
        p_lims = x2p_func([0,350])
        self.xaxis = []
        new_p_labels = []
        for x,y in zip(p_labels,E_labels):
            if np.isfinite(x):
                self.xaxis.append((x,y))
                new_p_labels.append(x)
        p_labels = new_p_labels
        # needed to be done in case the energy line is off the screen
        
        self.yaxis = [(0, self.y_mm[0]), (len(self.y_mm), self.y_mm[-1])]
        self.p_labels = p_labels
        
        # inverse of above - Ep = MeV_x_cal(xp)
        Mev_x_cal = interp1d(spec_x_mm, spec_MeV, kind='linear', copy=True, bounds_error=False, fill_value=np.nan)
        self.x_MeV = Mev_x_cal(self.x_mm)

        # to add
        #self.y_mrad = 

    def get_image(self,path):
        img_raw = self.get_raw_image(path)
        if hasattr(self, 'eSpec_proc'):
            img_pCpermm2 = self.eSpec_proc.espec_data2screen(img_raw)
            return img_pCpermm2

        else:
            # if something has failed, then default to raw img
            return img_raw

    def get_raw_image(self, path):
        img_raw = imread(path)
        return img_raw

    def get_total_charge(self,path):
        img_pCpermm2 = self.get_image(path)
        total_pC = np.sum(img_pCpermm2)*self.dx*self.dy
        print(total_pC)
        return total_pC

    def get_total_charge_from_im(self,img_raw):
        img_pCpermm2 = self.eSpec_proc.espec_data2screen(img_raw)
        total_pC = np.sum(img_pCpermm2)*self.dx*self.dy
        return total_pC
