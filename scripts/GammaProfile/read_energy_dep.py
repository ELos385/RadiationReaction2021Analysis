#read_energy_dep.py
import sys, os
sys.path.append('../../')
sys.path.append('../../..')
sys.path.append('../')
import scipy.io
import matplotlib.pyplot as plt
from setup import *
plt.style.use('/Users/ee.los/.matplotlib/mpl_configdir/thesis.mplstyle')

filename="/Users/ee.los/Documents/GitHub/RadiationReaction2021Analysis/calib/GammaProfile/GammaProfile_MonoGamma_Edep.mat"
mat = scipy.io.loadmat(filename)
print(mat.keys())
E_MeV=mat['Egamma_MeV']
E_dep_MeV=mat['edep_MeV']

plt.plot(E_MeV, E_dep_MeV)
plt.xlabel("Energy of incident photon (Mev)")
plt.ylabel("Energy Deposited (MeV)")
plt.xlim(0, 1)
plt.ylim(0, 0.2)
plt.tight_layout()
plt.show()
