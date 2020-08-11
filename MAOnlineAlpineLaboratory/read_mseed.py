import obspy
import numpy as np

channels = ["EHE.D", "EHN.D", "EHZ.D"]

for channel in channels:
    path_to_file = "/itet-stor/barthc/net_scratch/data/seismic_data/4D/MH36/2017/%s/" % channel
    filename_noerror = "4D.MH36.A.%s.20171122_020000.miniseed" % channel
    filename_error = "4D.MH36.A.%s.20171224_070000.miniseed" % channel
    st_noerror = obspy.read(path_to_file + filename_noerror, details=True)
    st_error = obspy.read(path_to_file + filename_error, details=True)

    print("NO ERROR FILE:\n", st_noerror)
    print(st_noerror[0].stats)
    print(st_noerror[0].data.shape)
    print(st_noerror[0].data)
    print("ERROR FILE:\n", st_error)
    print(st_error[0].stats)
    print(st_error[0].data.shape)
    print(st_error[0].data)

