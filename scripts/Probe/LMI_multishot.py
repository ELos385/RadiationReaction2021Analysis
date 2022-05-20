import sys
sys.path.append('../../')
from setup import *
from modules.Probe.interferometry import *
from lib.pipeline import *
from lib.general_tools import *
import numpy as np
import matplotlib.pyplot as plot

''' User inputs '''

# shot
date = '20210620'
run = 'Set1_nulls_20210620'
diag = 'LMI'

# reference
date_ref = '20210621'
run_ref = 'run06'
shot_num_ref = [1,2,3,5,6,7,8,9,10]

# regions of interest
fringes_roi = [220, 280, 155, 465] # vertical, horizontal
channel_roi = [35, 55, 30, 290]

retrieval_method = 'fft'  # hilbert or fft

um_per_px = 84.3

# mask for fft retrieval
mask_params = [113.0, 30.0, 16.0, 20.0, 8, 8]   #ux, uy, wx, wy, nx, ny

''' Run the code '''

I = interferometry(None)

I.fringes_roi = fringes_roi
I.channel_roi = channel_roi
I.umperpixel = um_per_px
I.fmask_params = mask_params
I.ref_shots = [[date_ref, run_ref, bs] for bs in shot_num_ref]

# use data pipeline
def LMI_processing(data):
    # function which runs get_ne but takes only the raw data as input
    # then produces a density lineout (but no error bars)
    density_map = I.get_ne_m(data)
    nrows, ncols = density_map.shape
    mid = nrows // 2

    channel_width = 8
    hw = channel_width // 2

    density_lineout = np.mean(density_map[mid-hw : mid+hw, :], axis = 0)

    return density_map #density_lineout

def LMI_processing_v2(data):
    fringes_img = I.get_fringe_image(data)

    nrows, ncols = fringes_img.shape
    u,v, = np.arange(ncols), np.arange(nrows)
    U,V = np.meshgrid(u,v)
    I.Gauss(U,V, I.fmask_params)

    ref_Z = I.get_ref_phase(I.ref_shots, method='fft')
    phase = I.calc_phase(fringes_img, ref_Z = ref_Z, method='fft')

    I.calc_channel_mask()
    phase_2 = I.correct_phase_polarity(phase, overwrite=False)
    phase_3 = I.phase_bg_cor(phase_2, overwrite=False)

    phase_4 = I.rotate_image(phase_3, I.calc_channel_angle(phase_3))

    centre = I.calc_phase_centre(phase_4,method=['convolution','gaussian'])
    int_ne_dl = I.phase_to_SI(phase_4,overwrite=False)
    density_map = I.abel_invert_int_ne_dl(int_ne_dl, method=None, overwrite=False)

    nrows, ncols = density_map.shape
    mid = nrows // 2

    channel_width = 8
    hw = channel_width // 2

    density_lineout = np.mean(density_map[mid-hw : mid+hw, :], axis = 0)

    return density_lineout

LMI_pipeline = DataPipeline(diag, LMI_processing_v2, single_shot_mode=True)
shot_num, LMI_data = LMI_pipeline.run('%s/%s'%(date, run))

print(LMI_data.shape)

x = np.arange(LMI_data.shape[1]) * I.umperpixel * 1e-3 # in mm


# write results to pkl files
for i in np.arange(len(shot_num)):
    result = {
        'description': 'LMI analysis',
        'date': date,
        'run': run,
        'shot': shot_num[i],
        'x axis': x,
        'density lineout': LMI_data[i]
    }
    save_object(result, 'C:/Users/ccct501/Documents/RadiationReaction2021Analysis/results/Probe/%s_%s_Shot%02s.pkl'%(diag,run,shot_num[i]))
