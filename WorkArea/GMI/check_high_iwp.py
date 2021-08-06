#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:01:17 2021

@author: inderpreet
"""


def high_iwp(iwp_mean):
    
    nanmask = np.isnan(iwp)
    a = iwp_mean[~nanmask] > 20
    if np.sum(a) > 2:
        print(iwp_mean[~nanmask].max())
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
import pandas as pd

path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/with_z0_he")
files = glob.glob(os.path.join(path, "1B*nc"))

gpath = os.path.expanduser("~/Dendrite/SatData/GMI/L1B/2020/01")
gfiles = glob.glob(os.path.join(gpath, "*", "*HDF5"))

inputs             = ["ta", "t2m",  "wvp", "lat", "stype", "z0"]
outputs            = "iwp"
batchSize          = 4
latlims            = [0, 65]

TB  = []
LAT = []
LON = []
IWP = []
LSM = []
T2M = []
WVP = []
Z0  = []

h_iwp_files = []
for ix, file in enumerate(files):
    print (ix)
    dataset = xarray.open_dataset(file)
    
    iwp = dataset.iwp_mean.data
    lat = dataset.lat.data
    iwp[np.abs(lat) > 64] = np.nan
    
    check = high_iwp(iwp)
    if check is True:
        h_iwp_files.append(file)
   
    dataset.close()     

with open("high_iwp_files_loglinear.pickle", "wb") as f:
    pickle.dump(h_iwp_files, f)    
    f.close()

for ix, file in enumerate(h_iwp_files[:]):
    print (ix)
       
    dataset = xarray.open_dataset(file)
    
    check = high_iwp(dataset.iwp_mean)
    
    if check is True:
        print (np.nanmax(dataset.iwp_mean)) 
        
        gfile = glob.glob(os.path.join(gpath, '*', os.path.basename(file)[:-3] + '*'))
        
        #print (os.path.basename(file))
        #print (os.path.basename(gfile[0]))
        #high_iwp_files.append(file)

        gmisat = GMI_Sat(gfile[0])
        # except:
        #     print("skipping this index, doing next file")
        #     continue

        validation_data    = gmiSatData(gmisat, 
                                  inputs, outputs,
                                  batch_size = batchSize,
                                  latlims = latlims,
                                  std = None,
                                  mean = None,
                                  log = None)
        mask = dataset.iwp_mean > 25.0
        print (np.sum(mask))
        #print(mask)
        TB.append(validation_data.x[:, :, :4][mask])
        LAT.append(validation_data.x[:, :, -3 ][mask])
        LON.append(validation_data.lon[:, :][mask])
        IWP.append(dataset.iwp_mean.data[mask])
        LSM.append(dataset.stype.data[mask])
        T2M.append(validation_data.x[:, :, 4][mask])
        WVP.append(validation_data.x[:, :, 5][mask])
        Z0.append(validation_data.x[:, :, -1][mask])
        
        dataset.close()

TB   = np.concatenate(TB, axis = 0)   
IWP  = np.concatenate(IWP, axis = 0)
LAT  = np.concatenate(LAT, axis = 0)
LON  = np.concatenate(LON, axis = 0)
LSM  = np.concatenate(LSM, axis = 0)
T2M  = np.concatenate(T2M, axis = 0)
WVP  = np.concatenate(WVP, axis = 0)
Z0   = np.concatenate(Z0, axis = 0)


d = xarray.Dataset({
    "tb": (["cases", "channels"], TB),
    "iwp": (["cases"], IWP),
    "lsm": (["cases"], LSM),
    "t2m": (["cases"], T2M),
    "wvp": (["cases"], WVP),
    "z0": (["cases"], Z0)},
    
coords={
    "lon": (["cases"], LON),
    "lat": (["scans"], LAT),
})

outfile = "high_iwp_inputs_greater_120.nc"
d.to_netcdf(outfile, mode = "w")
