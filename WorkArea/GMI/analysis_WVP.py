#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:09:30 2021

@author: inderpreet
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray
import os
import pickle
from iwc2tb.GMI.grid_field import grid_field  
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
plt.matplotlib.rcParams.update({'font.size': 20})
#%%
  
    
path = os.path.expanduser("~/git/Projects/IWC2TB/notebooks")
file_t0 = "/home/inderpreet/git/Projects/IWC2TB/notebooks/202001_WVP.nc"
file_t2 = "/home/inderpreet/git/Projects/IWC2TB/notebooks/202001_WVP_t2m.nc"


def regrid(file):
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
    
    return iwpg/counts, y
   
def statistics_wvp(y, y0):
     
     bias = np.mean(y - y0)
     rmse = np.sqrt(np.mean(y - y0) ** 2)
     mae  = np.mean(np.abs(y - y0))
     
     
     return bias, mae, rmse

y_t0, y0_t0      = regrid(file_t0)
y_t2, y0_t2      = regrid(file_t2)    
    



        
#%% read ERA5 TCWV
path    = os.path.expanduser("~/git/Projects/IWC2TB/WorkArea/GMI/")
erafile = os.path.join(path, "reanalysis-era5-single-levels-monthly-means_202001_total_column_water_vapour.nc")

dataset = xarray.open_dataset(erafile)

eralon = dataset.longitude.data
eralat = dataset.latitude.data
tcwv   = dataset.tcwv.data


lamask  = np.logical_or(np.abs(eralat) <= 45 , np.abs(eralat) >= 65) 
#%% plot histograms
bins = np.arange(0, 70, 0.25)
hist_t0, _ = np.histogram(y_t0, bins, density = True)
hist_t2, _ = np.histogram(y_t2, bins, density = True)
hist_er, _ = np.histogram(tcwv[:, lamask, :], bins, density = True)

bin_c = 0.5 * (bins[1:] + bins[:-1])


fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot(bin_c, hist_t0, label = "t0")
ax.plot(bin_c, hist_t2, label = "t2")
#ax.plot(bin_c, hist_er, label = "ERA")
ax.legend()
ax.set_xlabel("WVP [kg/m2]")
ax.set_ylabel("frequency")
ax.set_yscale("log")
fig.savefig("WVP_PDF.png", bbox_inches = "tight")

#%% statistics





#%%
eralon = eralon.reshape(-1, 1)
eralon = np.repeat(eralon, eralat.shape[0], axis = 1)
eralon = eralon.T

eralat = eralat.reshape(-1, 1)
eralat = np.repeat(eralat, tcwv.shape[2], axis = 1)

tcwvg, tcwvc = grid_field(eralat.ravel(), eralon.ravel(), tcwv.ravel(), gsize = 2.5, startlat = 65.0)
#%%

lats = np.arange(-65, 65, 2.5)
lons = np.arange(0, 360, 2.5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [20, 20])
   
ax1.pcolormesh(lons, lats, y_t0, vmin = 0, vmax = 25, cmap = cm.rainbow)
ax1.set_title("retrieved WVP")


latmask = np.abs(lats) <= 45
tcwvg[latmask] = np.nan

cs = ax2.pcolormesh(lons, lats, tcwvg/tcwvc, 
                    vmin = 0, vmax = 25, cmap = cm.rainbow)
ax2.set_title("ERA5")
fig.colorbar(cs, label="WVP [kg/m2]", ax = [ax1, ax2])
ax1.set_ylabel("latitude [deg]")
ax2.set_ylabel("latitude [deg]")
ax2.set_xlabel("longitude [deg]")

fig.savefig("WVP.png", bbox_inches = "tight")


#%% scatter plot monthly means

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.scatter(tcwvg/tcwvc, y_t0)
ax.set_xlabel("ERA5 TCWV [kg/m2] ")
ax.set_ylabel("Retrieved GMI TCWV [kg/m2] ")
x = np.arange(0, 30, 1)
y = x
ax.plot(x, y, 'k')
ax.set_title("Monthly means (January 2020)")

fig.savefig("WVP_scatter_monthlymean.png", bbox_inches = "tight")    