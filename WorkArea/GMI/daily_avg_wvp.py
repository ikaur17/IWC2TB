#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:02:37 2021

@author: inderpreet
"""

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import xarray
import os
import pickle
import glob
from iwc2tb.GMI.grid_field import grid_field  
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
plt.matplotlib.rcParams.update({'font.size': 20})

#%%

path     = os.path.expanduser("~/Dendrite/UserAreas/Kaur/WVP")
wvpfile =  os.path.join(path, "gridded_WVP_202001.nc") 

#%%

dataset = xarray.open_dataset(wvpfile)

wvp  = dataset.wvp_sum/dataset.wvp_counts

gmi_daily = np.nanmean(wvp, axis = (0, 1))


#%% ERA5 daily avg

erapath = "/home/inderpreet/git/Projects/pansat/notebooks/products/ERA5/reanalysis-era5-single-levels/"

startdate = datetime(2020, 1, 1, 0)

enddate = datetime(2020, 1, 15, 0)

tcwv_daily = np.zeros([721, 1440, 31])
ix = 0
while enddate >= startdate:
    
    
    tcwv = np.zeros([721, 1440])
    
    erafiles = glob.glob(os.path.join(erapath, "*" + startdate.strftime("%Y%m%d") + "*"))
    
    for erafile in erafiles:
        
        dataset = xarray.open_dataset(erafile)
        
        tcwv += dataset.tcwv.data[0, :, : ]
        
        dataset.close()
    
    
    tcwv = tcwv/len(erafiles)
    
    tcwv_daily[:, :, ix] = tcwv
    startdate += timedelta(days = 1)
    ix += 1

dataset = xarray.open_dataset(erafiles[0])
lat = dataset.latitude.data
lon = dataset.longitude.data 

mask = np.logical_and(np.abs(lat) >= 45.0, np.abs(lat) <= 65.0)

tcwv_daily = tcwv_daily[mask, :, :]

era_daily = np.mean(tcwv_daily, axis = (0, 1))

#%%
days = np.arange(1, 32, 1)
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot(days, gmi_daily, label = "GMI") 
ax.plot(days, era_daily, label = "ERA5")

ax.legend()  

fig.savefig("daily_mean_WVP.png", bbox_inches = "tight")

  
      
    
    
    
    

