#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:09:06 2021

@author: inderpreet
"""


import numpy as np
import xarray
import pickle
import matplotlib.pyplot as plt
from iwc2tb.GMI.grid_field import grid_field  
plt.rcParams.update({'font.size': 20})


def read_pickle(file):
    with open(file, "rb") as f:
          giwp  = pickle.load(f)
          giwp_mean  = pickle.load(f)
          giwp0 = pickle.load(f)
          glon  = pickle.load(f)
          glat  = pickle.load(f)
          glsm  = pickle.load(f)
          
          f.close()
          
          return giwp, giwp_mean, giwp0, glon, glat, glsm
      
      
      
def read_nc(file):  
    dataset = xarray.open_dataset(file)
    
    giwp = dataset.IWP.data
    giwp_m = dataset.iwp_mean.data
    giwp0 = dataset.iwp0.data
    glon = dataset.lon.data
    glat = dataset.lat.data
    glsm = dataset.lsm.data
    dataset.close()
    return giwp, giwp_m, giwp0, glon, glat, glsm
 
def zonal_mean(lat, iwp, latbins):
    

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins    
         

file1 = "jan2020_IWP_he_log.nc"
file2 = "jan2020_IWP_he_loglinear.nc"
#file3 = ""

#iwp2, iwp_mean2, iwp02, lon2, lat2, lsm2 = read_pickle(file2)
#iwp3, iwp_mean3, iwp03, lon3, lat3, lsm3 = read_pickle(file3)

#iwp_mean2 = np.concatenate(iwp_mean2, axis = 0)
#%%
iwp1, iwpm1, iwp01, lon1, lat1, lsm1 = read_nc(file1)
iwp2, iwpm2, iwp02, lon2, lat2, lsm2  = read_nc(file2)

#%% spare ice data
with open("spareice_jan2020.pickle", "rb") as f:
    slat = pickle.load(f)
    slon = pickle.load(f)
    siwp = pickle.load(f)
f.close()
smask = np.abs(slat) <= 65.0
siwpg, siwpc =  grid_field(slat[smask], slon[smask]%360, siwp[smask],
                                  gsize = 1.0, startlat = 65.0)

#%% dardar data 
with open("dardar_jan2009.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

dmask = np.abs(dlat) <= 65.0 
diwpg, diwpc =  grid_field(dlat[dmask], dlon[dmask]%360, diwp[dmask],
                                  gsize = 2.5, startlat = 65.0)

#%%

bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200, 300, 500,  1000, 2000])


cbins = (bins[1:] + bins[:-1])/2
fig, ax = plt.subplots(1, 1, figsize = [8, 8])

h1, _ = np.histogram(iwpm1, bins, density = True)
h2, _ = np.histogram(iwpm2, bins, density = True)
shist, _ = np.histogram(siwp * 0.001, bins, density = True)
dhist, _ = np.histogram(diwp, bins, density = True)

ax.plot(cbins, h1, label = "Log")
ax.plot(cbins, h2, label = "LogLinear")
ax.plot(cbins, shist, label = "SpareIce")
ax.plot(cbins, dhist, label = "DARDAR")

ax.set_yscale("log")
ax.set_xscale("log")

ax.legend()

ax.set_xlabel("IWP [kg/m2]")
ax.set_ylabel("PDF")                                                                                                                                                        

fig.savefig("Figures/PDF_diff_versions_jan.png", bbox_inches = "tight")

#%%

nanmask = ~np.isnan(iwpm1)
latbins = np.arange(-65, 66, 1.5)

ziwp1, ziwpc1 = zonal_mean(lat1[nanmask], iwpm1[nanmask], latbins)

nanmask = ~np.isnan(iwpm2)
ziwp2, ziwpc2 = zonal_mean(lat2[nanmask], iwpm2[nanmask], latbins)


ziwp_si, ziwp_sic = zonal_mean(slat, siwp, latbins)
ziwp_d, ziwp_dc = zonal_mean(dlat,  diwp, latbins)


fig, ax = plt.subplots(1, 1, figsize = [15, 15])
ax.plot(ziwp1[:-1]/ziwpc1[:-1],latbins, 'b-',  label = "Log") 


#ax.plot(ziwp/ziwpc,latbins, 'b-',  label = "QRNN") 
ax.plot(ziwp2[:-1]/ziwpc2[:-1], latbins, 'r.-', label = "LogLinear")
#ax.plot(ziwp0/ziwp0c, latbins, 'b.-', label = "GPROF")

ax.plot( (0.001 * ziwp_si/ziwp_sic), latbins, 'b--', label = "SpareIce")
ax.plot(ziwp_d/ziwp_dc,latbins, 'b:', label = "DAR|DAR") 

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.legend()
fig.savefig("Figures/zonal_meandiff_versions_jan.png", bbox_inches = "tight")

#%%