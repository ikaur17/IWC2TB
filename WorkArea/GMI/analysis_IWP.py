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
import os
plt.rcParams.update({'font.size': 30})



#%%

def zonal_mean(lat, iwp, latbins):
    

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins


def histogram(iwp, bins):
    
    hist, _ = np.histogram(iwp, bins)
    
    return hist/np.sum(hist)


def plot_hist(siwp, diwp, giwp, giwp0, slat, dlat, glat, latlims ): 
    
    
    bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200])
    
    smask = np.logical_and(np.abs(slat) >= latlims[0] , np.abs(slat) <= latlims[1])
    dmask = np.logical_and(np.abs(dlat) >= latlims[0] , np.abs(dlat) <= latlims[1])
    gmask = np.logical_and(np.abs(glat) >= latlims[0] , np.abs(glat) <= latlims[1])   

    shist = histogram(0.001 * siwp[smask], bins)

    dhist = histogram(diwp[dmask], bins)
    ghist = histogram(giwp[gmask], bins)
    ghist0 = histogram(giwp0[gmask], bins)    
    
    bin_center = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    
    ax.plot(bin_center, shist, 'o-', label = "SI" )
    ax.plot(bin_center, dhist, 'o-', label = "DARDAR" )
    ax.plot(bin_center, ghist, 'o-',label = "GMI QRNN" )
    ax.plot(bin_center, ghist0, 'o-',label = "GMI GPROF" )

    
    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(str(latlims[0]) +"-" +str(latlims[1]))
    fig.savefig("Figures/PDF_IWP_" +str(latlims[0]) +"-" +str(latlims[1]) + ".png", bbox_inches = "tight")

#%%
with open("spareice_jan2020.pickle", "rb") as f:
    slat = pickle.load(f)
    slon = pickle.load(f)
    siwp = pickle.load(f)
f.close()
smask = np.abs(slat) <= 65.0
siwpg, siwpc =  grid_field(slat[smask], slon[smask]%360, siwp[smask],
                                  gsize = 1.0, startlat = 65.0)

#%%
with open("dardar_jan2009.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

dmask = np.abs(dlat) <= 65.0 
diwpg, diwpc =  grid_field(dlat[dmask], dlon[dmask]%360, diwp[dmask],
                                  gsize = 1.0, startlat = 65.0)

#%%
janfile = os.path.expanduser("jan2020_IWP.pickle")
with open(janfile, "rb") as f:
    giwp  = pickle.load(f)
    giwp0 = pickle.load(f)
    glon  = pickle.load(f)
    glat  = pickle.load(f)
    glsm  = pickle.load(f)
    
    f.close()
    

#%%

nanmask = ~np.isnan(giwp)
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
giwpg, giwpc =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360, 
                           giwp[gmask].ravel(),
                           gsize = 1.0, startlat = 65.0)   

#%%
nanmask = giwp0 >  -9000
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
giwp0g, giwp0c =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360, 
                           giwp0[gmask].ravel(),
                           gsize = 1.0, startlat = 65.0)    





#%% plot histograms  

lsmmask = np.ones(giwp.shape, dtype = "bool")  

#lsmmask = glsm == 0

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], slat, dlat, glat[lsmmask], latlims )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], slat, dlat, glat[lsmmask], latlims )

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], slat, dlat, glat[lsmmask], latlims )  
    
#%% plot zonal_means
nanmask = np.logical_and(giwp0 > -9000, lsmmask)

latbins = np.arange(-65, 66, 1.5)
ziwp, ziwpc = zonal_mean(glat[nanmask], giwp[nanmask], latbins)
ziwp0, ziwp0c = zonal_mean(glat[nanmask], giwp0[nanmask], latbins)

ziwp_si, ziwp_sic = zonal_mean(slat, siwp, latbins)
ziwp_d, ziwp_dc = zonal_mean(dlat,  diwp, latbins)

fig, ax = plt.subplots(1, 1, figsize = [15, 15])
ax.plot(ziwp[:-1]/ziwpc[:-1],latbins, '.-',  label = "QRNN") 
ax.plot(ziwp0[:-1]/ziwp0c[:-1], latbins, '.-', label = "GPROF")

ax.plot( (0.001 * ziwp_si/ziwp_sic), latbins, '.-', label = "SpareIce")
ax.plot(ziwp_d/ziwp_dc,latbins, '.-', label = "DARDAR") 

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.legend()
fig.savefig("Figures/zonal_mean_all.png", bbox_inches = "tight")

#%% plot zonal_means from gridded data

lats = np.arange(-65, 65, 1.0)
gziwp_s = np.mean(siwpg/siwpc, axis = 1)
gziwp_d = np.mean(diwpg/diwpc, axis = 1)
gziwp_g = np.mean(giwpg/giwpc, axis = 1)
gziwp_g0 = np.mean(giwp0g/giwp0c, axis = 1)


fig, ax = plt.subplots(1, 1, figsize = [15, 15])

ax.plot(0.001 * gziwp_s, lats, label = "SI")
ax.plot(gziwp_d, lats, label = "DARDAR")
ax.plot(gziwp_g, lats, label = "QRNN")
ax.plot(gziwp_g0, lats, label = "GPROF")
ax.legend()

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")

fig.savefig("Figures/zonal_mean_gridded.png", bbox_inches = "tight")

  
#%% get avg IWP, weighted by cosine of latitude [g/m2]

lats  = np.arange(-65, 65, 1)
cosines = np.cos(np.deg2rad(lats))


print ("SI mean: ", np.sum(gziwp_s * cosines)/np.sum(cosines))
print ("DARDAR mean: ", np.sum(gziwp_d * cosines)/np.sum(cosines))
print ("QRNN mean: ", 1000 * np.sum(gziwp_g * cosines)/np.sum(cosines)) # g/m2
print ("GPROF mean: ", np.sum(gziwp_g0 * cosines)/np.sum(cosines))

#%% spatial distribution
lon = np.arange(0, 360, 1)
lat = np.arange(-65, 65, 1)

fig, axes = plt.subplots(4, 1, figsize = [40, 20])
fig.tight_layout()
ax = axes.ravel()
ax[0].pcolormesh(lon, lat, 0.001 * siwpg/siwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 50),  cmap = cm.rainbow)
ax[1].pcolormesh(lon, lat, diwpg/diwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 50),  cmap = cm.rainbow)
cs = ax[2].pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-3, vmax= 50),  cmap = cm.rainbow)
cs = ax[3].pcolormesh(lon, lat, giwp0g/giwp0c, norm=colors.LogNorm(vmin=1e-3, vmax= 50),  cmap = cm.rainbow)
fig.colorbar(cs, label="IWP [kg/m2]", ax = axes)

ax[0].set_title("SpareICE")
ax[1].set_title("DARDAR")
ax[2].set_title("GMI QRNN")
ax[3].set_title("GPROF")
fig.savefig("Figures/IWP_spatial_distribution.png", bbox_inches = "tight")