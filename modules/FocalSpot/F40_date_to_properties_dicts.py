#F2_date_to_properties_dicts.py

#FWHM temporal: april 8th 2021
# 43fs
# 1st June 2021: somewhere around 100 fs
# 2021-06-06: 43-45fs
# 2021-06-08: 150fs (this was done the night before whilst dazzler scanning)
# 2021-06-08: 80-120fs, north & south
# 2021-06-09: 53 fs ish
# NORTH ENERGY Averaged over 10 shots from 286236 to 286245
# 2021-06-09:
# Pump: 46.13 J
# Uncomp: 11.09 J
#
# NORTH PL Averaged over 5 shots from 286236 to 286240
# 54.95 fs
#
# SOUTH ENERGY Averaged over 9 shots from 286236 to 286245
# Pump: 46.77 J
# Uncomp: 11.66 J
#
# SOUTH PL Averaged over 5 shots from 286237 to 286242
# 59.52 fs
#dazzler higher order settings reset to 03/04 before shooting on 09/06/2021: 43-45fs?? Not measured.
# 2021-06-10: 55-60fs

date_to_burst_and_position={
20210607:{2:12.0, 3:10.0, 4:8.0, 5:6.0, 6:4.0, 7:14.0, 8:16.0, 9:18.0, 10:20.0},
20210608:{4:23.0, 5:25.0, 6:27.0, 7:29.0, 8:21.0, 9:19.0, 10:17.0}
}

F40_date_to_run_dict={
20210607:'ref01',
20210608:'ref01'
}

F40_date_to_FWHM_t_dict={
20210607:[44.0*10**-15, 1.0*10**-15],
20210608:[44.0*10**-15, 1.0*10**-15],
}

#these are approximate: from ecat2
F40_date_to_south_energy_dict={
20210607:[14.0, 0.01],
20210608:[9.0, 0.01]
}

#throughput: april 8th 2021
#North: 61.2573% from before compressor into target chamber (air)
#North: 47.2222% including effect of pellicle & holey oap
# get final numbers from Matt: not in lab book for some reason.
F40_date_to_south_thoughput_dict={#including pellicle and holey f2 parabola
20210607:[0.59, 0.01],
20210608:[0.59, 0.01]
}


#magnification: lens objective
#north: x 50

#north reworked and updated, 21-04-21

#2021-04-23 x 10 objective

#2021-05-25 x 20 objective

#2021-06-08 north calibration image
#(jetx45594_0019
# using Chris A's spatial calibration for now;
# 0.50+-0.01 microns per pixel

F40_date_to_spatial_calibration_dict={
20210607:[1.0037, 0.0256],
20210608:[1.0037, 0.0256]
}
