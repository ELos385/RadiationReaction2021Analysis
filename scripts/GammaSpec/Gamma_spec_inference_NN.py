#GammaSpec_fit_Ec.py
"""
GammaStack Image processing and spectrum fitting for Compton data,
handles one shot at a time.
"""

import os, sys
sys.path.append('../../')
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, leastsq
from scipy.special import kv, kn, expi
from scipy.stats import loguniform
from scipy.constants import e, c, epsilon_0, m_e, Boltzmann, hbar, Planck
import math
import matplotlib
import matplotlib.pyplot as plt
import cv2
import emcee
import corner
from scipy.ndimage import median_filter, rotate
from scipy.io import loadmat
from skimage.io import imread
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from setup import *
from lib.general_tools import *
from lib.pipeline import *
from modules.GammaSpec.GammaSpecProc import *

diag='CsIStackTop'#'CsIStackTop'#'CsIStackSide'

# date='20210620'
# run='run12'
# shot='Shot001'

#load crystal properties from mat file
coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg=load_crystal_properties(diag)
#load energy deposition info from mat file
Egamma_MeV_interp, CsIEnergy_ProjZ_interp=load_Edep_info(diag)
#load correction factor for CsI stack
corr_factor_mean, corr_factor_se=load_correction_factor(diag)

CsIStack=GammaStack(coords, N_crystals_X, N_crystals_Y, pos_array_X, pos_array_Y, crystal_size_XY_pxl, rot_deg, Egamma_MeV_interp, CsIEnergy_ProjZ_interp, corr_factor_mean, corr_factor_se, kernel=3, debug=False)
#CsIStack.plot_contours(GammaStack_Img)


A=0.15
Ec=40.0
N_E=10000
E=np.logspace(-1, 2.3, N_E)
# dN_dE_inv_compt=A*(E/Ec)**(-2.0/3.0)*np.exp(E/Ec)

A_arr=np.array([0.15, 0.6, 0.34])
Ec_arr=np.array([35.0, 23.0, 46.5])

def dN_dE_inv_compt(E, Ec, A):
    return A*(E/Ec)**(-2.0/3.0)*np.exp(-E/Ec)

def calc_Brems_energy_spec(E, Te_MeV, B):
    gff=3.0**0.5/np.pi*np.exp(E/(2.0*Te_MeV))*kn(0, E/(2.0*Te_MeV))#expi(0.5*(E/Te_MeV)**2)
    dP_dE_brehms=B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)*gff
    # print("gff=%s"%gff)
    # print("B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)=%s"%(B*1.0/(Te_MeV)**0.5*np.exp(-E/Te_MeV)))
    return dP_dE_brehms

def gaussian_spec(E, mean, sigma, height):
    return height*np.exp(-(E-mean)**2/(2.0*sigma**2))

def generate_test_and_training_data(Inv_Compton_ranges, N_Brem_ranges, N_Gauss_ranges, N_Inv_Compton, N_Brem, N_Gauss, N_multi_spec, CsIStack):
    N_total=N_Inv_Compton+N_Brem+N_Gauss+N_multi_spec
    E=CsIStack.Egamma_MeV_interp#np.logspace(-3, 2.3, 50)
    #CsIStack.Egamma_MeV_interp
    output_spectra=np.zeros((N_total, len(E)))
    output_signal=np.zeros((N_total, CsIStack.N_crystals_X))

    A_sampled=loguniform(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1]).rvs(size=N_Inv_Compton)#np.random.uniform(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1], N_Inv_Compton)
    Ec_sampled=loguniform(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1]).rvs(size=N_Inv_Compton)#np.random.uniform(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1], N_Inv_Compton)#loguniform(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1]).rvs(size=N_Inv_Compton)
    # A_sampled=np.random.uniform(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1], N_Inv_Compton)#np.random.uniform(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1], N_Inv_Compton)
    # Ec_sampled=np.random.uniform(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1], N_Inv_Compton)
    # for i in range(N_Inv_Compton):
    #     output_spectra[i, :]=dN_dE_inv_compt(E, Ec_sampled[i], A_sampled[i])
    #     for k in range(0, CsIStack.N_crystals_X):
    #         output_signal[i, k]=np.trapz(output_spectra[i, :]*CsIStack.CsIEnergy_ProjZ_interp[:, k], E)
    #     # output_signal[i]=output_signal[i]/np.mean(output_signal[i])

    # a_sampled=np.linspace(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1], N_Inv_Compton)
    # ec_sampled=np.linspace(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1], N_Inv_Compton)
    # A_sampled, Ec_sampled=np.meshgrid(a_sampled, ec_sampled)
    # A_sampled=A_sampled.flatten()
    # Ec_sampled=Ec_sampled.flatten()
    ########
    B_sampled=loguniform(N_Brem_ranges[0, 0], N_Brem_ranges[0, 1]).rvs(size=N_Brem)
    Te_sampled=loguniform(N_Brem_ranges[1, 0], N_Brem_ranges[1, 1]).rvs(size=N_Brem)

    #np.array([[5.0, 50.0],[1.0*10**-3, 5.0]])
    gauss_mean_sampled=loguniform(N_Gauss_ranges[0, 0], N_Gauss_ranges[0, 1]).rvs(size=N_Gauss)
    gauss_height_sampled=loguniform(N_Gauss_ranges[1, 0], N_Gauss_ranges[1, 1]).rvs(size=N_Gauss)
    gauss_std_sampled=np.zeros(N_Gauss)

    No_spectra_summed=np.random.randint(2, 10, N_multi_spec)
    E_integrate=np.linspace(min(CsIStack.Egamma_MeV_interp), max(CsIStack.Egamma_MeV_interp), 5000)
    for i in range(N_total):
        if i<N_Inv_Compton:
            output_spectra[i, :]=dN_dE_inv_compt(E, Ec_sampled[i], A_sampled[i])
            for k in range(0, CsIStack.N_crystals_X):
                output_signal[i, k]=np.trapz(dN_dE_inv_compt(E_integrate, Ec_sampled[i], A_sampled[i])*np.interp(E_integrate, E, CsIStack.CsIEnergy_ProjZ_interp[:, k]), E_integrate)#np.trapz(output_spectra[i, :]*CsIStack.CsIEnergy_ProjZ_interp[:, k], E)
        #output_signal[i, :]=output_signal[i, :]/(max(output_signal[i, :])-min(output_signal[i, :]))
        elif i >=N_Inv_Compton and i<N_Brem+N_Inv_Compton:
            output_spectra[i, :]=calc_Brems_energy_spec(E, Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton])
            for k in range(0, CsIStack.N_crystals_X):
                output_signal[i, k]=np.trapz(calc_Brems_energy_spec(E_integrate, Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton])*np.interp(E_integrate, E, CsIStack.CsIEnergy_ProjZ_interp[:, k]), E_integrate)
            # if np.isnan(np.sum(output_spectra[i, :])):
            #     print(output_spectra[i, 100])
            #     plt.scatter(Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton], color='b')
            #     print("out spec=%s, Te_sampled[i-N_Inv_Compton]=%s, B_sampled[i-N_Inv_Compton]=%s"%(output_spectra[i, 0], Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton]))
            # else:
            #     plt.scatter(Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton], color='r')
            # print(Te_sampled[i-N_Inv_Compton])
            # print(B_sampled[i-N_Inv_Compton])
            # print(calc_Brems_energy_spec(E, Te_sampled[i-N_Inv_Compton], B_sampled[i-N_Inv_Compton]))
        elif i >=N_Brem+N_Inv_Compton and i<N_Brem+N_Inv_Compton+N_Gauss:
            gauss_std_sampled_temp=loguniform(gauss_mean_sampled[i-(N_Brem+N_Inv_Compton)]/100.0, gauss_mean_sampled[i-(N_Brem+N_Inv_Compton)]/3.0).rvs(size=1)
            gauss_std_sampled[i-(N_Brem+N_Inv_Compton)]=gauss_std_sampled_temp
            output_spectra[i, :]=gaussian_spec(E, gauss_mean_sampled[i-(N_Brem+N_Inv_Compton)], gauss_std_sampled[i-(N_Brem+N_Inv_Compton)], gauss_height_sampled[i-(N_Brem+N_Inv_Compton)])
            for k in range(0, CsIStack.N_crystals_X):
                output_signal[i, k]=np.trapz(gaussian_spec(E_integrate, gauss_mean_sampled[i-(N_Brem+N_Inv_Compton)], gauss_std_sampled[i-(N_Brem+N_Inv_Compton)], gauss_height_sampled[i-(N_Brem+N_Inv_Compton)])*np.interp(E_integrate, E, CsIStack.CsIEnergy_ProjZ_interp[:, k]), E_integrate)
        # else:
        #     summed_spectra=0
        #     spectra_type=np.random.randint(1, 3, No_spectra_summed[i-(N_Brem+N_Inv_Compton+N_multi_spec)])
        #     for j in range(No_spectra_summed[i-(N_Brem+N_Inv_Compton+N_multi_spec)]):
        #         if spectra_type[j]==1:
        #             Ec_sampled_multi_spec=loguniform(Inv_Compton_ranges[1, 0], Inv_Compton_ranges[1, 1]).rvs(size=1)
        #             A_sampled_multi_spec=loguniform(Inv_Compton_ranges[0, 0], Inv_Compton_ranges[0, 1]).rvs(size=1)
        #             summed_spectra+=dN_dE_inv_compt(E, Ec_sampled_multi_spec, A_sampled_multi_spec)
        #         elif spectra_type[j]==2:
        #             B_sampled_multi_spec=loguniform(N_Brem_ranges[0, 0], N_Brem_ranges[0, 1]).rvs(size=1)
        #             Te_sampled_multi_spec=loguniform(N_Brem_ranges[1, 0], N_Brem_ranges[1, 1]).rvs(size=1)
        #             summed_spectra+=calc_Brems_energy_spec(E, Te_sampled_multi_spec, B_sampled_multi_spec)
        #         elif spectra_type[j]==3:
        #             gauss_mean_multi_spec=loguniform(N_Gauss_ranges[0, 0], N_Gauss_ranges[0, 1]).rvs(size=1)
        #             gauss_height_multi_spec=loguniform(N_Gauss_ranges[1, 0], N_Gauss_ranges[1, 1]).rvs(size=1)
        #             gauss_std_multi_spec=loguniform(gauss_mean_multi_spec/100.0, gauss_mean_multi_spec/3.0)
        #             summed_spectra+=gaussian_spec(E, gauss_mean_multi_spec, gauss_std_multi_spec, gauss_height_multi_spec)
        #     output_spectra[i, :]=summed_spectra
        # for k in range(0, CsIStack.N_crystals_X):
        #     output_signal[i, k]=np.trapz(output_spectra[i, :]*CsIStack.CsIEnergy_ProjZ_interp[:, k], E)
        # output_signal[i, :]=output_signal[i, :]/(max(output_signal[i, :])-min(output_signal[i, :]))
    # plt.show()
    return output_spectra, output_signal, Ec_sampled, A_sampled#, Te_sampled, B_sampled, gauss_mean_sampled, gauss_std_sampled, gauss_height_sampled#, output_signal,

def build_and_compile_cnn_model(layer3_nodes, layer4_nodes, layer5_nodes):#, layer5_nodes
    actv='relu'#'sigmoid'#'relu'
    actv2=keras.layers.LeakyReLU(alpha=0.3)#'sigmoid'
    layer1_nodes=14#CsIStack.N_crystals_X
    layer7_nodes=250#len(CsIStack.Egamma_MeV_interp)
    model = tf.keras.Sequential([
    # good for Brems and ICS
    # E has 120 points, train with dataset of 5000
    # tf.keras.layers.Dense(CsIStack.N_crystals_X, input_dim=CsIStack.N_crystals_X, activation=actv),
    # tf.keras.layers.Dense(2, activation=actv),
    # tf.keras.layers.Dense(5, activation=actv),
    # tf.keras.layers.Dense(25, activation=actv),
    # tf.keras.layers.Dense(len(CsIStack.Egamma_MeV_interp), activation=actv2)#'linear'
    tf.keras.layers.Dense(layer1_nodes, input_dim=layer1_nodes, activation=actv),
    # tf.keras.layers.Dense(5, activation=actv),
    tf.keras.layers.Dense(3, activation=actv),
    tf.keras.layers.Dense(layer3_nodes, activation=actv),
    tf.keras.layers.Dense(layer4_nodes, activation=actv),
    # tf.keras.layers.Dense(35, activation=actv),
    tf.keras.layers.Dense(layer5_nodes, activation=actv),
    # tf.keras.layers.Dense(layer6_nodes, activation=actv),
    tf.keras.layers.Dense(layer7_nodes, activation=actv2)#'linear'
    # CsIStack.N_crystals_X, 0, 3, 35, 0, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 5, 75, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 25, 45, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 15, 65, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 25, 65, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 5, 85, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 25, 85, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 35, 95, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 55, 85, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 65, 85, len(CsIStack.Egamma_MeV_interp),
    ######### CsIStack.N_crystals_X, 0, 3, 75, 85, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 5, 95, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 3, 65, 95, len(CsIStack.Egamma_MeV_interp),# lowest loss
    # CsIStack.N_crystals_X, 0, 3, 75, 95, len(CsIStack.Egamma_MeV_interp),

    # CsIStack.N_crystals_X, 7, 3, 15, 75, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 7, 2, 15, 75, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 2, 25, 0, len(CsIStack.Egamma_MeV_interp),
    # CsIStack.N_crystals_X, 0, 2, 35, 0, len(CsIStack.Egamma_MeV_interp), ## best so far
    # CsIStack.N_crystals_X, 0, 2, 45, 0, len(CsIStack.Egamma_MeV_interp), # similar to 25
    # CsIStack.N_crystals_X, 0, 2, 55, 0, len(CsIStack.Egamma_MeV_interp), # similar to 25
    # CsIStack.N_crystals_X, 0, 2, 10, 35, len(CsIStack.Egamma_MeV_interp), # similar to 25
    # CsIStack.N_crystals_X, 0, 2, 20, 45, len(CsIStack.Egamma_MeV_interp)
    # CsIStack.N_crystals_X, 0, 2, 15, 85, len(CsIStack.Egamma_MeV_interp)
    ])
    opt=keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=keras.losses.LogCosh(),#
    optimizer=opt,
    # optimizer= keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    metrics=['acc'])
    # print(model.metrics_names)
    return model

N_train_inv_compton=0000#8000#5000#50000##8000
N_train_brems=0000#50000#0#200
N_train_gaussians=10000#500000#200
N_train_multi_spec=0#400
N_train=N_train_inv_compton+N_train_brems+N_train_gaussians+N_train_multi_spec

N_test_inv_compton=000#100#100
N_test_brems=000#100#20
N_test_gaussians=100#100#20
N_test_multi_spec=0#40
N_test=N_test_inv_compton+N_test_brems+N_test_gaussians+N_test_multi_spec

# N_inv_compton=N_train_inv_compton+N_test_inv_compton
# N_brems=N_train_brems+N_test_brems
# N_gaussians=N_train_gaussians+N_test_gaussians
# N_multi_spec=N_train_multi_spec+N_test_multi_spec
N_total=N_train+N_test
# Remeber when training for real data to use large ranges, to avoid the normalisation excluding test points (or include test data in normed array)


Inv_Compton_ranges=np.array([[0.05, 1.0],[10.0, 100.0]])
N_Brem_ranges=np.array([[9.0*10**-3, 1.0*10**-2],[0.705, 5.0]])#np.array([[5.0*10**-4, 1.0*10**-2],[0.05, 50.0]])
N_Gauss_ranges=np.array([[7.0, 50.0],[1.0*10**-3, 5.0]])

#output_spectra, output_signal, Ec_sampled, A_sampled, Te_sampled, B_sampled, gauss_mean_sampled, gauss_std_sampled, gauss_height_sampled=generate_test_and_training_data(Inv_Compton_ranges, N_Brem_ranges, N_Gauss_ranges, N_inv_compton, N_brems, N_gaussians, N_multi_spec, CsIStack)
# test_spectra, test_output_signal, test_Ec_sampled, test_A_sampled, test_Te_sampled, test_B_sampled, test_gauss_mean_sampled, test_gauss_std_sampled, test_gauss_height_sampled=generate_test_and_training_data(Inv_Compton_ranges, N_Brem_ranges, N_Gauss_ranges, N_test_inv_compton, N_test_brems, N_test_gaussians, N_test_multi_spec, CsIStack)
training_spectra, output_signal, Ec_sampled, A_sampled=generate_test_and_training_data(Inv_Compton_ranges, N_Brem_ranges, N_Gauss_ranges, N_train_inv_compton, N_train_brems, N_train_gaussians, N_train_multi_spec, CsIStack)
test_spectra, test_output_signal, test_Ec_sampled, test_A_sampled=generate_test_and_training_data(Inv_Compton_ranges, N_Brem_ranges, N_Gauss_ranges, N_test_inv_compton, N_test_brems, N_test_gaussians, N_test_multi_spec, CsIStack)

print(output_signal.shape)
print(test_spectra.shape)
print(test_output_signal.shape)
print(training_spectra.shape)
# for i in range():
#     plt.scatter(CsIStack.Egamma_MeV_interp, training_spectra[i, :], color=color_arr[i], marker='.')
crystal_arr=np.linspace(1, len(output_signal[0]), len(output_signal[0]))
# print(crystal_arr)
# for i in range(N_train_inv_compton):
#     plt.plot(crystal_arr, output_signal[i])
# plt.show()

# print(np.log10(training_spectra[0, :]))

n=N_test_inv_compton

full_output_signal=np.concatenate((output_signal, test_output_signal)).reshape(N_total, -1)
full_spectra=np.concatenate((training_spectra, test_spectra)).reshape(N_total, -1)

# print(full_output_signal.shape)
# print(full_spectra.shape)
# c='r'
# for i in range(len(full_output_signal)):
#     if i>N_train:
#         c="b"
#     #plt.plot(crystal_arr, full_output_signal[i, :], color=c, marker='.')
#     plt.plot(CsIStack.Egamma_MeV_interp, full_spectra[i, :], color=c, marker='.')
# plt.show()

# scaler_output_signal_train = preprocessing.StandardScaler(with_mean=False).fit(full_output_signal)#MinMaxScaler(feature_range=(0, 1))
scaler_spectra_train =preprocessing.StandardScaler(with_mean=False).fit(full_spectra)#MaxAbsScaler()
scaler_output_signal_train = preprocessing.StandardScaler(with_mean=False).fit(full_output_signal)#MinMaxScaler(feature_range=(0, 1))
#scaler_spectra_train =preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(full_spectra)
# scaler_spectra_train =preprocessing.StandardScaler(with_mean=False).fit(full_spectra)#MaxAbsScaler()
# scaler_output_signal_train = preprocessing.StandardScaler(with_mean=False).fit(output_signal)#MinMaxScaler(feature_range=(0, 1))
# scaler_spectra_train = preprocessing.StandardScaler(with_mean=False).fit(training_spectra)

# normalised_output_signal=scaler_output_signal_train.transform(output_signal)#.reshape(1, N_train, CsIStack.N_crystals_X)
# normalized_test_output_signal = scaler_output_signal_train.transform(test_output_signal)#3.reshape(1, N_test, CsIStack.N_crystals_X)

normalised_output_signal=scaler_output_signal_train.transform(output_signal)
normalized_test_output_signal = scaler_output_signal_train.transform(test_output_signal)
normalized_training_spectra = scaler_spectra_train.transform(training_spectra)#.reshape(1, N_train, len(CsIStack.Egamma_MeV_interp))
normalized_test_spectra = scaler_spectra_train.transform(test_spectra)#.reshape(1, N_test, len(CsIStack.Egamma_MeV_interp))

# no_workers=4
# batch_size=2000
# Dtrain=tf.data.Dataset.from_tensor_slices((normalised_output_signal, normalized_training_spectra)).shuffle(N_train_inv_compton*3).repeat().batch(no_workers*batch_size)
# #Dtrain_spec=tf.data.Dataset.from_tensor_slices((normalized_training_spectra)).shuffle(N_train_inv_compton*3).repeat().batch(no_workers*batch_size)
# Dtest=tf.data.Dataset.from_tensor_slices((normalized_test_output_signal, normalized_test_spectra))#.shuffle(N_train_inv_compton*3).repeat().batch(no_workers*batch_size)
# # tf.enable_eager_execution()
# for x, y in Dtrain:
#     print(x.shape)
#     print(y.shape)
# print(Dtrain.shape)
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA#AutoShardPolicy.OFF
# Dtrain = Dtrain.with_options(options)
# Dtest = Dtest.with_options(options)
# inverse transform and print

# compile the keras model
# 'mean_squared_logarithmic_error'
# name="mean_absolute_error"
# opt = keras.optimizers.Adam(learning_rate=0.005)#'categorical_crossentropy'
# # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# # tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
#
# model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
#     model=build_and_compile_cnn_model(CsIStack)
# model.summary()
model=build_and_compile_cnn_model(45, 45, 125)#
history=model.fit(normalised_output_signal, normalized_training_spectra, batch_size=16, epochs=300, shuffle=True, verbose=1)#, validation_data=(normalized_test_output_signal, normalized_test_spectra))#verbose=0 for no output#epochs=10, batch_size=10
#layer5_nodes=[0, 45, 65, 85, 105, 125]

# param_grid=dict(layer3_nodes=[5, 25, 45, 65], layer4_nodes=[25, 45, 65, 85, 105, 125], layer5_nodes=[85, 105, 125, 145])#165, 185, 205, 225
# model =  KerasClassifier(build_fn=build_and_compile_cnn_model, batch_size=16, epochs=250, verbose = False, shuffle=True)
# gsh = HalvingGridSearchCV(estimator=model, param_grid=param_grid)#, factor=2, resource ='n_samples', min_resources=3)
# grid_result=gsh.fit(normalised_output_signal, normalized_training_spectra)#, validation_data=(normalized_test_output_signal, normalized_test_spectra))#verbose=0 for no output#epochs=10, batch_size=10
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# results = pd.DataFrame.from_dict(grid_result.cv_results_)
# print(results)

#validation_data=(Dtest)
# evaluate the keras model
_, accuracy = model.evaluate(normalised_output_signal, normalized_training_spectra)
print('Accuracy: %.2f' % (accuracy))
# predictions = model.predict(normalised_output_signal)#model.predict(test_output_signal[0:2, :])

# Plot loss and validation loss
# loss = np.array(history.history["loss"])
# # val_loss = np.array(history.history["val_loss"])
#
# fig, ax = plt.subplots(1,1,figsize=(6,2),dpi=150)
# ax.semilogy(loss / loss[0],  label="loss")
# # ax.semilogy(val_loss / val_loss[0], label="val_loss")
# ax.set_xlabel("Epoch")
# ax.legend()
# plt.show()

predictions = model.predict(normalized_test_output_signal)

inversed_predictions = scaler_spectra_train.inverse_transform(predictions)

print(predictions)
print(predictions.shape)

color_arr=['r', 'b', 'orange', 'g', 'cyan', 'magenta', 'lightgreen', 'black', 'purple', 'pink']

for i in range(5):
    plt.scatter(CsIStack.Egamma_MeV_interp, predictions[i, :], color=color_arr[i], alpha=0.5, marker='.')
    # plt.plot(CsIStack.Egamma_MeV_interp, training_spectra[i, :], color=color_arr[i], marker='.')
    # plt.plot(CsIStack.Egamma_MeV_interp, normalized_test_spectra[i, :], color=color_arr[i])
    plt.plot(CsIStack.Egamma_MeV_interp, normalized_test_spectra[i, :], color=color_arr[i])
plt.xlim(0, 200.0)
plt.show()


for i in range(5):
    plt.scatter(CsIStack.Egamma_MeV_interp, inversed_predictions[i, :], color=color_arr[i], alpha=0.5, marker='.')
    # plt.plot(CsIStack.Egamma_MeV_interp, training_spectra[i, :], color=color_arr[i], marker='.')
    plt.plot(CsIStack.Egamma_MeV_interp, test_spectra[i, :], color=color_arr[i])
# plt.yscale('log')
# plt.ylim(10**-5, 1.0)
plt.xlim(0, 500.0)
plt.show()

# for i in range(n):
#     plt.scatter(CsIStack.Egamma_MeV_interp, predictions[i, :], color=color_arr[i], alpha=0.5, marker='.')
#     plt.scatter(CsIStack.Egamma_MeV_interp, test_spectra[i, :], color=color_arr[i],  marker='.')
# plt.yscale('log')
# plt.xlim(0, 180.0)
# plt.show()

print(inversed_predictions.shape)
print(test_spectra.shape)

prediction_loss=1.0/n*np.sum((inversed_predictions-test_spectra)**2/test_spectra[0:n, :], axis=1)*100
no_predictions=np.linspace(0, n, n)
# plt.scatter(no_predictions, prediction_loss)
plt.hist(prediction_loss, bins=20)
plt.show()



# plot least sqares


# print("Ec=%s, A=%s"%(Ec_sampled[5], A_sampled[5]))
# print("Te=%s, B=%s"%(Te_sampled[27-20], B_sampled[27-20]))
# print("mean=%s, std=%s, height=%s"%(gauss_mean_sampled[53-40], gauss_std_sampled[53-40], gauss_height_sampled[53-40]))
#
# plt.plot(CsIStack.Egamma_MeV_interp, training_spectra[5, :], color='g')
# plt.plot(CsIStack.Egamma_MeV_interp, training_spectra[27, :], color='orange')
# plt.plot(CsIStack.Egamma_MeV_interp, training_spectra[53, :], color='r')
# # plt.yscale('log')
# plt.show()
#
# crystals_X_arr=np.linspace(1, len(output_signal[0]), len(output_signal[0]))
#
# plt.plot(crystals_X_arr, output_signal[5, :], color='g')
# plt.plot(crystals_X_arr, output_signal[27, :], color='orange')
# plt.plot(crystals_X_arr, output_signal[53, :], color='r')
# plt.show()
