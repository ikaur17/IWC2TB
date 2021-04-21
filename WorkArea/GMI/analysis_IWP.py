#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:35:20 2021

@author: inderpreet
"""


import numpy as np
import pickle
from iwc2tb.GMI.grid_field import grid_field  
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
#%%


def zonal_mean(lat, iwp, latbins):
    

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins


def histogram(iwp, bins):
    
    hist, _ = np.histogram(iwp, bins)
    
    return hist/np.sum(hist)

#%%
with open("spareice.pickle", "rb") as f:
    slat = pickle.load(f)
    slon = pickle.load(f)
    siwp = pickle.load(f)
f.close()
smask = np.abs(slat) <= 30.0
siwpg, siwpc =  grid_field(slat[smask], slon[smask]%360, siwp[smask],
                                  gsize = 1.0, startlat = 30.0)

with open("dardar.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

dmask = np.abs(dlat) <= 30.0 
diwpg, diwpc =  grid_field(dlat[dmask], dlon[dmask]%360, diwp[dmask],
                                  gsize = 1.0, startlat = 30.0)

 
with open("gridded_iwp.pickle", "rb") as f:
    giwpg   = pickle.load(f)
    giwpc   = pickle.load(f)
    giwpg0  = pickle.load(f)
    giwpc0  = pickle.load(f)

    f.close()

giwpg = np.sum(giwpg, axis = 0)
giwpc = np.sum(giwpc, axis = 0)

giwpg0 = np.sum(giwpg0, axis = 0)
giwpc0 = np.sum(giwpc0, axis = 0)

file = "/home/inderpreet/Dendrite/UserAreas/Kaur/IWP/202001_IWP_t2m.nc"
with open(file[:-3] +"_ghist.pickle", "rb") as f:
    ghist_t2m = pickle.load(f)
f.close()    

file = "/home/inderpreet/Dendrite/UserAreas/Kaur/IWP/202001_IWP_wvp.nc"
with open(file[:-3] +"_ghist.pickle", "rb") as f:
    ghist_wvp = pickle.load(f)
f.close()     


ghist_wvp = np.sum(ghist_wvp, axis = 0)   
ghist_wvp = ghist_wvp/np.sum(ghist_wvp)


ghist_t2m = np.sum(ghist_t2m, axis = 0)   
ghist_t2m = ghist_t2m/np.sum(ghist_t2m)

#%%
bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200])


shist = histogram(0.001 * siwp[smask], bins)
dhist = histogram(diwp[dmask], bins)


bin_center = 0.5 * (bins[1:] + bins[:-1])
fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.plot(bin_center, shist, 'o-', label = "SI" )
ax.plot(bin_center, dhist, 'o-', label = "DARDAR" )
ax.plot(bin_center, ghist_t2m, 'o-',label = "GMI t2m" )
ax.plot(bin_center, ghist_wvp, 'o-',label = "GMI t2m + wvp" )

ax.set_xlabel("IWP [kg/m2]")
ax.set_ylabel("frequency")
ax.legend()
ax.set_yscale("log")
ax.set_xscale("log")

fig.savefig("PDF_IWP.png", bbox_inches = "tight")


#%%
lon = np.arange(0, 360, 1)
lat = np.arange(-30, 30, 1)

fig, axes = plt.subplots(4, 1, figsize = [60, 10])
fig.tight_layout()
ax = axes.ravel()
ax[0].pcolormesh(lon, lat, 0.001 * siwpg/siwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 15),  cmap = cm.rainbow)
ax[1].pcolormesh(lon, lat, diwpg/diwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 15),  cmap = cm.rainbow)
cs = ax[2].pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 15),  cmap = cm.rainbow)
cs = ax[3].pcolormesh(lon, lat, giwpg0/giwpc0, norm=colors.LogNorm(vmin=1e-3, vmax= 15),  cmap = cm.rainbow)
fig.colorbar(cs, label="IWP [kg/m2]", ax = axes)

ax[0].set_title("SpareICE")
ax[1].set_title("DARDAR")
ax[2].set_title("GMI QRNN")
ax[3].set_title("GPROF")
fig.savefig("IWP_spatial_distribution.png", bbox_inches = "tight")