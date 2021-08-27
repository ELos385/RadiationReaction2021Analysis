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

dict={
20210603:{2:42132, 3:42147, 4:42162, 5:42177, 6:42207, 7:42237, 8:42237, 9:42117, 10:42102, 11:42087, 12:42057, 13:42057, 14:42027, 15:41878, 16:41650, 17:42385},
20210604:{1:41564, 2:41589, 3:41614, 4:41614, 5:41639, 6:41664, 7:41539, 8:41514, 9:41489, 10:41464},
20210614:{3:40989, 4:41004, 5:41019, 6:41034, 7:41069, 8:40974, 9:40959, 10:40944, 11:40909},
20210617:{1:41460, 2:41475, 3:41490, 4:41505, 5:41540, 6:41445, 7:41430, 8:41415, 9:41380},
20210618:{1:41481, 2:41496, 3:41511, 4:41526, 5:41561, 6:41481, 7:41466, 8:41451, 9:41436, 10:41401},
20210620:{1:28144, 2:28159, 3:28174, 4:28189, 5:28129, 6:28114, 7:28099, 8:28099, 9:28114, 10:28129, 11:28144},
20210621:{1:41487, 2:41502, 3:41517, 4:41532, 5:41547, 6:41562, 7:41577, 8:41592, 9:41607, 10:41540},
20210622:{1:28287, 2:28287, 3:28302, 4:28317, 5:28332, 6:28347, 7:28362, 8:28377, 9:28392, 10:28407, 11:28422, 12:28437, 13:28340}
}

F2_date_to_run_dict={
20210603:'ref02',
20210604:'ref01',
20210614:'ref01',
20210617:'ref04',
20210618:'ref01',
20210620:'ref01',
20210621:'run01',
20210622:'ref02'
}

F2_date_to_FWHM_t_dict={
20210603:[44.0*10**-15, 1.0*10**-15],
20210604:[44.0*10**-15, 1.0*10**-15],
20210614:[57.5*10**-15, 2.5*10**-15],
20210617:[57.5*10**-15, 2.5*10**-15],
20210618:[57.5*10**-15, 2.5*10**-15],
20210620:[57.5*10**-15, 2.5*10**-15],
20210621:[57.5*10**-15, 2.5*10**-15],
20210622:[57.5*10**-15, 2.5*10**-15]
}

#these are approximate: from ecat2
F2_date_to_north_energy_dict={
20210603:[8.0, 0.01],
20210604:[12.0, 0.01],
20210614:[12.0, 0.01],
20210617:[11.0, 0.01],
20210618:[12.5, 0.01],
20210620:[13.5, 0.01],
20210621:[11.0, 0.01],
20210622:[7.5, 0.01]
}

#throughput: april 8th 2021
#North: 61.2573% from before compressor into target chamber (air)
#North: 47.2222% including effect of pellicle & holey oap
# get final numbers from Matt: not in lab book for some reason.
F2_date_to_north_thoughput_dict={#including pellicle and holey f2 parabola
20210603:[0.472222, 0.01],#error??
20210604:[0.472222, 0.01],
20210614:[0.472222, 0.01],
20210617:[0.472222, 0.01],
20210618:[0.472222, 0.01],
20210620:[0.472222, 0.01],
20210621:[0.472222, 0.01],
20210622:[0.472222, 0.01]
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

F2_date_to_spatial_calibration_dict={
20210603:[0.50, 0.01],
20210604:[0.50, 0.01],
20210614:[0.50, 0.01],
20210617:[0.50, 0.01],
20210618:[0.50, 0.01],
20210620:[0.50, 0.01],
20210621:[0.50, 0.01],
20210622:[0.50, 0.01]
}
