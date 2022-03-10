#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:33:48 2021

@author: inderpreet
"""
import xarray
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%
def zonal_mean(lat, iwp, latbins):


    bins     = np.digitize(lat, latbins)

    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)

    return iwp_mean, nbins


inputfile = "jan2017_IWP_lpa1.nc"
dataset = xarray.open_dataset(inputfile)
giwp_mean = dataset.iwp_mean.data
giwp0 = dataset.iwp0.data
glon = dataset.lon.data
glat = dataset.lat.data
dataset.close()

inputfile = "2017_y95.nc"
dataset = xarray.open_dataset(inputfile)
y95 = dataset.y95.data
dataset.close()



inputfile = "2017_y5.nc"
dataset = xarray.open_dataset(inputfile)
y5 = dataset.y5.data
dataset.close()
#%%

with open("dardar_jan2017.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()


#%% plot zonal_means
plt.rcParams.update({'font.size': 18})
latbins = np.arange(-66, 65, 2.5)


ziwp_d, ziwp_dc = zonal_mean(dlat,  diwp, latbins)


giwp = giwp_mean
nanmask = giwp0 > -9000

ziwp0, ziwp0c = zonal_mean(glat[nanmask], giwp0[nanmask], latbins)


nanmask = ~np.isnan(giwp)
#nanmask = ~nanmask & (glsm == 0)
ziwp, ziwpc = zonal_mean(glat[nanmask], giwp[nanmask], latbins)

nanmask = ~np.isnan(y5)
ziwp_5, ziwpc_5 = zonal_mean(glat[nanmask], y5[nanmask], latbins)

nanmask = ~np.isnan(y95)
ziwp_95, ziwpc_95 = zonal_mean(glat[nanmask], y95[nanmask], latbins)



#%%

fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.plot(ziwp[:-1]/ziwpc[:-1],latbins, 'r-', linewidth = 2.5, label = r"Q-IWP")

ax.plot(ziwp0[:-1]/ziwp0c[:-1], latbins, 'k', linewidth = 2.5, label = "GPROF")


ax.fill_betweenx( latbins, ziwp_95[:-1]/ziwpc_95[:-1], ziwp_5[:-1]/ziwpc_5[:-1],
                 alpha = 0.2 )

#ax.plot(ziwp_5[:-1]/ziwpc_5[:-1],latbins, 'r-',  label = r"")

ax.plot(ziwp_d[:-1]/ziwp_dc[:-1],latbins, 'b-', linewidth = 2.5, label = "DARDAR")

#ax.plot(ziwp_95[:-1]/ziwpc_95[:-1],latbins, 'r-',  label = r"QRNN (99.0 kg m$^{-2}$)")


ax.set_ylabel("Latitude [deg]")
ax.set_xlabel(r"IWP [kg m$^{-2}$]")
ax.legend()
ax.grid("on", alpha = 0.3)
fig.savefig("Figures/zonal_mean_jan_2017.pdf", bbox_inches = "tight")
