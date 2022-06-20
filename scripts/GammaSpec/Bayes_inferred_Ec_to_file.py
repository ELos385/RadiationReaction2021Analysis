#Bayes_inferred_Ec_to_file.py

import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv, kn, expi
from scipy.constants import e, c, epsilon_0, m_e, Boltzmann, hbar, Planck
import math
import matplotlib
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from skimage.io import imread
from schwimmbad import MPIPool

from lib.general_tools import *
from lib.pipeline import *
from modules.GammaSpec.GammaSpecProc import *

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Bayesian inference set-up
    Bayes_guess=np.array([30.0, 0.05, 0.05])
    no_dim=4
    no_walkers=(no_dim+1)*2
    no_steps=10000
    no_burn=8000
    err_guess=1000.0
    percent_std=np.array([0.1])

    base_dir_Top=ROOT_DATA_FOLDER+'CsIStackTop'
    base_dir_Side=ROOT_DATA_FOLDER+'CsIStackSide'
    date='20210604'

    # for date in os.listdir(base_dir):
    #     if date[0]=='2':
    CsIStackSide=initialise_gamma_stack_obj('CsIStackSide', date)
    CsIStackTop=initialise_gamma_stack_obj('CsIStackTop', date)

    guess=CsIStackTop.generate_estimates(Bayes_guess, err_guess, percent_std, no_walkers, no_dim)

    GammaSpec_results_dict={}
    for run in os.listdir(base_dir_Top+'/'+date+'/'):
        if run[0:2]=='ru':# and int(run[3:])>5
            run_no=int(run[3:])
            run_path_Top=base_dir_Top+'/'+date+'/'+run
            run_path_Side=base_dir_Side+'/'+date+'/'+run

            DataPipeline_CsIStackSide=DataPipeline('CsIStackSide', CsIStackSide.get_measured_signal_summed_over_columns, single_shot_mode=True)
            DataPipeline_CsIStackTop=DataPipeline('CsIStackTop', CsIStackTop.get_measured_signal_summed_over_columns, single_shot_mode=True)
            print(run_path_Top)
            shot_number, CsIStackSide_props=DataPipeline_CsIStackSide.run(run_path_Side)
            shot_number, CsIStackTop_props=DataPipeline_CsIStackTop.run(run_path_Top)
            shot_number=np.array(shot_number).astype(int)
            print(CsIStackSide_props)
            #GammaSpec_results_dict[run]={}
            GammaSpec_results_dict={}
            for i in range(0, len(shot_number)):
                popt=Bayes_mcmc(guess, no_walkers, no_steps, no_burn, no_dim, CsIStackTop_props[i], CsIStackSide_props[i], CsIStackTop, CsIStackSide, pool)
                print(i)
                # show corner plot
                # fig = corner.corner(popt)#flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)])
                # plt.show()

                # CsIStackSide.plot_bayes_inferred_spec(popt)
                # arr_out=CsIStackTop.plot_bayes_inferred_spec(popt)
                # # plt.plot(CsIStack.Egamma_MeV_interp,dN_dE_test)
                # plt.show()
                #
                # CsIStackSide.plot_transmission_inferred(popt, measured_signal_summed_over_columns_Side)
                #
                # CsIStackTop.plot_transmission_inferred(popt[:, 0:no_dim-1], measured_signal_summed_over_columns_Top)
                # plt.show()
                #GammaSpec_results_dict[run][shot_number[i]]=popt
                GammaSpec_results_dict[shot_number[i]]=popt
            out_filepath='/Users/ee.los/Documents/GitHub/RadiationReaction2021Analysis/results/GammaSpec/%s/'%(date)
            if not os.path.exists(out_filepath):
                os.makedirs(out_filepath)
            filename=out_filepath+'GammaStack_%s.pkl'%(run)
            save_object(GammaSpec_results_dict, filename)

    # results_in=load_object(filename)
    # print(results_in.keys())
    # run_key_arr=list(results_in.keys())
    #
    # shot_key_arr=list(results_in[run_key_arr[0]].keys())
    # plot_params=results_in[run_key_arr[0]][shot_key_arr[0]]
    # print(plot_params/CsIStackSide.norm_factor)
    # # print(results_in.keys())
    # # print(results_in[])
    #
    # CsIStackSide.plot_bayes_inferred_spec(plot_params)
    # arr_out=CsIStackTop.plot_bayes_inferred_spec(plot_params)
    # # plt.plot(CsIStack.Egamma_MeV_interp,dN_dE_test)
    # plt.show()
    #
    # CsIStackSide.plot_transmission_inferred(plot_params, CsIStackSide_props[0], Invert=False)
    # CsIStackTop.plot_transmission_inferred(plot_params, CsIStackTop_props[0], Invert=True)
    # plt.show()
