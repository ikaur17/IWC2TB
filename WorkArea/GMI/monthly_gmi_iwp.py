#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:01:17 2021

@author: inderpreet
"""


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

path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/")
files = glob.glob(os.path.join(path, "1B*202002*nc"))

gpath = os.path.expanduser("~/Dendrite/SatData/GMI/L1B/2020/02")
gfiles = glob.glob(os.path.join(gpath, "*", "*HDF5"))

inputs             = ["ta", "t2m",  "wvp", "lat", "stype"]
outputs            = "iwp"
batchSize          = 4
latlims            = [0, 65]


LAT = []
LON = []
IWP = []
LSM = []
IWP0 = []
for ix, file in enumerate(files[:]):
    print (ix)
    
    dataset = xarray.open_dataset(file)
    
    LAT.append(dataset.lat.data)
    LON.append(dataset.lon.data)
    IWP.append(dataset.iwp_mean.data)
    LSM.append(dataset.stype.data)
    
    dataset.close()
    
    gmisat = GMI_Sat(gfiles[ix])
    
    validation_data    = gmiSatData(gmisat, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             std = None,
                             mean = None,
                             log = None)
    
    IWP0.append(validation_data.y)

IWP  = np.concatenate(IWP, axis = 0)
LAT  = np.concatenate(LAT, axis = 0)
LON  = np.concatenate(LON, axis = 0)
LSM  = np.concatenate(LSM, axis = 0)
IWP0 = np.concatenate(IWP0, axis = 0)

with open("feb2020_IWP.pickle", "wb") as f:
    pickle.dump(IWP, f)
    pickle.dump(IWP0, f)
    pickle.dump(LON, f)
    pickle.dump(LAT, f)
    pickle.dump(LSM, f)
    
    f.close()




