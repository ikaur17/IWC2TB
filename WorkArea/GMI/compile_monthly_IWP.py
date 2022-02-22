#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:01:17 2021

@author: inderpreet
"""


def check_high_iwp(iwp_mean):
    
    a = iwp_mean > 120
    if np.sum(a) > 0:
        return True
    else:
        return False
    
import xarray
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from iwc2tb.GMI.gmiSatData import gmiSatData
import pickle
import time


#---------------------------------
year  = "2020"
month = "01"
#---------------------------------

path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/")
files = glob.glob(os.path.join(path, "1B*nc"))

gpath = os.path.expanduser(os.path.join("~/Dendrite/SatData/GMI/L1B/", year, month))
gfiles = glob.glob(os.path.join(gpath, "*", "*HDF5"))

inputs             = ["ta", "t2m",  "wvp", "lat", "stype"]
outputs            = "iwp"
batchSize          = 4
latlims            = [0, 65]


LAT = np.zeros([1, 221])
LON = np.zeros([1, 221])
IWP = np.zeros([1, 221])
IWP_mean = np.zeros([1, 221])
LSM = np.zeros([1, 221])
IWP0 = np.zeros([1, 221])

for ix, file in enumerate(files[:]):
    print (ix)
    
    try:
        gmisat = GMI_Sat(gfiles[ix])
    except:
        print("skipping this index, doing next file")
        continue
        

    
    
    dataset = xarray.open_dataset(file)
    
    LAT = np.concatenate(LAT, dataset.lat.data, axis = 0)
    LON = np.concatenate(LON, dataset.lon.data, axis = 0)
    IWP_mean = np.concatenate(IWP_mean, dataset.iwp_mean.data, axis = 0)
    IWP = np.concatenate(IWP, dataset.iwp.data, axis = 0)
    LSM = np.concatenate(LSM, dataset.stype.data, axis = 0)

    
    
    dataset.close()
    

    validation_data    = gmiSatData(gmisat, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims)
    
    IWP0 = np.concatenate(IWP0, validation_data.y)


# with open("jan2020_IWP_adam_normal.pickle", "wb") as f:
#     pickle.dump(IWP, f)
#     pickle.dump(IWP_mean, f)
#     pickle.dump(IWP0, f)
#     pickle.dump(LON, f)
#     pickle.dump(LAT, f)
#     pickle.dump(LSM, f)
    
#     f.close()


d = xarray.Dataset({
    "IWP": (["pixels", "scans"], IWP),
    "iwp_mean": (["pixels", "scans"], IWP_mean),
    "iwp0": (["pixels", "scans"], IWP0),
    "lsm": (["pixels", "scans"], LSM)} ,   
coords={
    "lon": (["pixels", "scans"], LON),
    "lat": (["pixels", "scans"], LAT),
})

outfile = "jan2020_IWP_adam.nc"
d.to_netcdf(outfile, mode = "w")



