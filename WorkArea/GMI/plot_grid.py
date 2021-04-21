#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:44:35 2021

@author: inderpreet
"""

import numpy as np
import xarray
import os
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from iwc2tb.GMI.grid_field import grid_field
#%%

  
    
path = os.path.expanduser("~/git/Projects/IWC2TB/notebooks")
files = ["/home/inderpreet/git/Projects/IWC2TB/notebooks/202001_WVP.nc"]


for file in files:
    print(file)
    dataset = xarray.open_dataset(file)
    
    lon = dataset.lon.data
    lat = dataset.lat.data
    y   = dataset.y_mean.data
    
    dataset.close()
    iwpg, counts = grid_field(lat, lon, y)
    
    outfile = file[:-3] + "_regridded.pickle"
    
    with open(outfile, 'wb') as f:
        
        pickle.dump(iwpg, f)
        pickle.dump(counts, f)
   
    f.close()
    
#%%   

        
#%% read ERA5 TCWV
path    = os.path.expanduser("~/git/Projects/IWC2TB/WorkArea/GMI/")
erafile = os.path.join(path, "reanalysis-era5-single-levels-monthly-means_202001_total_column_water_vapour.nc")

dataset = xarray.open_dataset(erafile)

eralon = dataset.longitude.data
eralat = dataset.latitude.data
tcwv   = dataset.tcwv.data


lamask  = np.logical_or(np.abs(eralat) <= 45 , np.abs(eralat) >= 65)

#%%

lats = np.arange(-65, 65, 2.5)
lons = np.arange(0, 360, 2.5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [20, 20])
   
ax1.pcolormesh(lons, lats, iwpg/counts, vmin = 0, vmax = 25, cmap = cm.rainbow)
ax1.set_title("retrieved WVP")
tcwvg = tcwv.copy()

tcwvg[0, lamask, :]  = np.nan
hmask =  np.abs(eralat) >= 65
cs = ax2.pcolormesh(eralon[::10], eralat[~hmask][::10], tcwvg[ 0, ~hmask, ::10][::10], 
                    vmin = 0, vmax = 25, cmap = cm.rainbow)
ax2.set_title("ERA5")
fig.colorbar(cs, label="WVP [kg/m2]", ax = [ax1, ax2])
ax1.set_ylabel("latitude [deg]")
ax2.set_ylabel("latitude [deg]")
ax2.set_xlabel("longitude [deg]")

fig.savefig("WVP.png", bbox_inches = "tight")






















        
        
            
        
        