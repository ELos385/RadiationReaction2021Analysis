#focal_scan_checker.py


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

#%%
date='20210527'
run= 'run14'


# f40 settings
f40_mid_y, f40_mid_x = 387, 364
f40_hw = 30

#probe settings
p_mid_y, p_mid_x = 236, 531
p_hw = 30



fp_head = ROOT_DATA_FOLDER + '/' 
fp_tail = '/' + date + '/' + run +'/' + 'Shot'


focal_spots = []
probes = []
for i in range(45):
    idx = i + 1
    print(i)
    s = str(idx)
    s = '0'*(3-len(s)) + s
    
    # default output if failed
    f40_img = np.full((2*f40_hw, 2*f40_hw), fill_value=np.nan)

    try:
        fn = fp_head + 'F40Leakage' + fp_tail + s + '.tiff'
        f40_img = imread(fn)
        f40_img = f40_img[f40_mid_y-f40_hw : f40_mid_y + f40_hw, f40_mid_x-f40_hw : f40_mid_x + f40_hw]
    except(FileNotFoundError):
        pass

    
    try:
        fn = fp_head + 'LMS' + fp_tail + s + '.tiff'
        probe_img = imread(fn)
        probe_img = probe_img[p_mid_y-p_hw : p_mid_y + p_hw, p_mid_x-p_hw : p_mid_x + p_hw]
    except(FileNotFoundError):
        pass
    
    focal_spots.append(f40_img)
    probes.append(probe_img)
    
#%%
    
# try quick focal spot analysis
    
focal_stats = []
for im in focal_spots:
    #im = leakage_spots[16]
    im = np.array(im, dtype=float)
    
    im -= np.median(im) # bkg correct
    
    im[im<0] = 0
    
    
    nrows, ncols = im.shape
    x, y =np.arange(ncols), np.arange(nrows)
    X,Y = np.meshgrid(x,y)
    centroid = [np.sum(im*X)/np.sum(im), np.sum(im*Y)/np.sum(im)]
    
    xl = np.sum(im, axis=0)
    x0 = np.sum(x * xl) / np.sum(xl)
    yl = np.sum(im, axis=1)
    y0 = np.sum(y * yl) / np.sum(yl)
    
    ox = np.abs(np.nansum(xl * (x - x0)**2) / np.nansum(xl))**(0.5)
    oy = np.abs(np.nansum(yl * (y - y0)**2) / np.nansum(yl))**(0.5)
    
    X_ids = (X>=x0-ox) & (X<=x0+ox)
    Y_ids = (Y>=y0-oy) & (Y<=y0+oy)
    frac = np.sum(im * X_ids * Y_ids) / np.sum(im)
    # looks to be a good metric!
    
    o = (ox*oy)**(0.5)
    
    plt.figure()
    plt.imshow(im, origin='lower')
    plt.plot(x0, y0, 'ro')
    plt.axvline(x0 + ox), plt.axvline(x0 - ox)
    plt.axhline(y0 + oy), plt.axhline(y0 - oy)
    focal_stats.append([x0, y0, ox, oy, o, frac])

focal_stats = np.array(focal_stats)

#%%

# check shock isn't doing anything funny
shocks_pos = []

for im in probes:
    #im = shocks[14]
    
    nrows, ncols = im.shape
    x, y =np.arange(ncols), np.arange(nrows)
    X,Y = np.meshgrid(x,y)
    centroid = [np.sum(im*X)/np.sum(im), np.sum(im*Y)/np.sum(im)]
    
    channel_height = 30
    channel_width = 5
    
    data = np.mean( im[channel_height-channel_width:channel_height+channel_width, :], axis=0)
    data = np.gradient(data)
    nidx = np.argmin(data[23:35]) + 23
    
    #plt.figure()
    #plt.imshow(im)
    #plt.axvline(x=nidx)
    #shocks_pos.append(nidx)

shocks_pos = np.array(shocks_pos)

#%%
data_path = '/Volumes/Backup2/2021_Radiation_Reaction/Automation/Outputs/focus_scan_20210527_run14.txt'

f_pos = np.loadtxt(data_path, skiprows=1, dtype=str, delimiter=' ')

f_pos = np.delete(f_pos, 0, 0)
f_pos = np.delete(f_pos, 0, 0)
shot_nums = np.array(f_pos[:,1], dtype=int)
shot_nums[8] = 9
shot_nums[17] = 18

f_pos = np.array(f_pos[:,-1], dtype=float)

plt.figure()
plt.scatter(f_pos, focal_stats[:,-1], marker='o')
plt.xlabel('ao focal term')
plt.ylabel('F40Leakage frac')

plt.figure()
plt.scatter(f_pos, shocks_pos, marker='o')
plt.xlabel('ao focal term')
plt.ylabel('shock pos on probe')

