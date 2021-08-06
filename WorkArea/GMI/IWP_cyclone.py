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


LAT = []
LON = []
IWP = []
IWP_mean = []
LSM = []
IWP0 = []
for ix, file in enumerate(files[:]):
    print (ix)
    
    try:
        gmisat = GMI_Sat(gfiles[ix])
    except:
        print("skipping this index, doing next file")
        continue
        

    
    
    dataset = xarray.open_dataset(file)
    
    LAT.append(dataset.lat.data)
    LON.append(dataset.lon.data)
    IWP_mean.append(dataset.iwp_mean.data)
    IWP.append(dataset.iwp.data)
    LSM.append(dataset.stype.data)
    
    dataset.close()
    

    validation_data    = gmiSatData(gmisat, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims)
    
    IWP0.append(validation_data.y)

IWP  = np.concatenate(IWP, axis = 0)
IWP_mean = np.concatenate(IWP_mean, axis = 0)
LAT  = np.concatenate(LAT, axis = 0)
LON  = np.concatenate(LON, axis = 0)
LSM  = np.concatenate(LSM, axis = 0)
IWP0 = np.concatenate(IWP0, axis = 0)

with open("jan2020_IWP_adam_normal.pickle", "wb") as f:
    pickle.dump(IWP, f)
    pickle.dump(IWP_mean, f)
    pickle.dump(IWP0, f)
    pickle.dump(LON, f)
    pickle.dump(LAT, f)
    pickle.dump(LSM, f)
    
    f.close()




