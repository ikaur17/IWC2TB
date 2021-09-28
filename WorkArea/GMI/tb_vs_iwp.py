#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 21:36:52 2021

@author: inderpreet
"""

import numpy as np
import xarray
import os
from iwc2tb.GMI.gmiSatData import gmiSatData
from iwc2tb.GMI.GMI_SatData import GMI_Sat
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
from iwc2tb.GMI.three_sigma_rule import three_sigma
import time
plt.rcParams.update({'font.size': 20})



#%%
def plot_pd(allmask, mask, ax):

    tbmask = tb[:, :, 3][allmask] < mask   #  cloudy cases
    

    tb0     = tb[:, :, 0][allmask][tbmask]
    pd      = (tb[:, :, 0] - tb[:, :, 1])[allmask][tbmask]
    iwp0    = giwp[allmask][tbmask]
    
    if tb0.shape[0] > 1000000:
        tb0 = tb0[::50]
        pd = pd[::50]
        iwp0 = iwp0[::50]
        
    cs = ax[0].scatter(tb0, pd, c = iwp0, s = 1, 
                  norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
    
    tb0     = tb[:, :, 0][allmask][~tbmask]
    pd      = (tb[:, :, 0] - tb[:, :, 1])[allmask][~tbmask]
    iwp0    = giwp[allmask][~tbmask]
    
    if tb0.shape[0] > 1000000:
        tb0 = tb0[::50]
        pd = pd[::50]
        iwp0 = iwp0[::50]
    
    cs = ax[1].scatter(tb0, pd, c = iwp0, s = 1, 
                  norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
        
    fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", shrink = 0.8, ax = ax)

#%%
inputfile = "jan2020_IWP_lpa.nc"
tbfile = "jan2020_tb_lpa.nc"
  
#%%  
dataset1 = xarray.open_dataset(inputfile)

giwp = dataset1.iwp_mean.data
glon = dataset1.lon.data
glat = dataset1.lat.data
glsm = dataset1.lsm.data

giwp[giwp < 1e-4] = 0

dataset1.close()

#%%
dataset2 = xarray.open_dataset(tbfile)

tb = dataset2.tb.data

#%%
train = xarray.open_dataset("/home/inderpreet/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train.nc") 

tiwp = train.ta.iwp
tlat = train.ta.lat
tlon = train.ta.lon% 360
#%% check TB over Tibetan plateau

lamask = (glat > 25) & (glat < 45)
lomask = (glon > 70) & (glon < 95)
gmask = np.logical_and(lamask, lomask)

lamask = (tlat > 25) & (tlat < 45)
lomask = (tlon > 70) & (tlon < 95)
tmask = np.logical_and(lamask, lomask)

bins = np.arange(100, 310, 1)
bin_center = (bins[1:] + bins[:-1])/2

fig, ax = plt.subplots(1,1 , figsize = [8, 8])
ax.hist(tb[:, :, 3][gmask], bins, histtype = "step", density = True )
ax.hist(train.ta.data[:, 3][tmask], bins, histtype = "step", density = True )
ax.set_yscale("log")

#%%

fig, ax = plt.subplots(2, 1, figsize = [8 , 16])
ax = ax.ravel()
pd = (tb[:, :, 0] - tb[:, :, 1])[gmask]
ax[0].scatter(tb[:, :, 0][gmask], pd)
ax[0].set_xlim([100, 320])
ax[0].set_ylim([-10, 50])
pd = (train.ta.data[:, 0] - train.ta.data[:, 1])[tmask]
ax[1].scatter(train.ta.data[:, 0][tmask], pd)
ax[1].set_xlim([100, 320])
ax[1].set_ylim([-10, 70])


#%%
#fig, ax = plt.subplots(1, 1, figsize = [8, 8])
#bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5,7,8, 9, 10,12, 14, 16, 20])
#ax.scatter(aro.iwp_mean.data[mask][::50], tro.iwp_mean.data[mask][::50])
#ax.legend()

#%%
isnow = glsm == 2
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.scatter(tb[:, :, -1][isnow], giwp[isnow])

#%% S.H. high lats

latmask = glat < -45

fig, ax = plt.subplots(2, 2, figsize = [15, 15])
#fig.tight_layout(pad = 0.5)
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")

ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r" > 45$^{\circ}$S")
fig.savefig("Figures/jul_sh_high_lat_iwp.png", bbox_inches = "tight", dpi = 300 )
#%% S.H. subTropics

latmask = (glat > -45) & (glat < -30) 
    
fig, ax = plt.subplots(2, 2, figsize = [15, 15])
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")
ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r"30$^{\circ}$S - 45$^{\circ}$S ")
fig.savefig("Figures/jul_sh_sub_trop_iwp.png", bbox_inches = "tight", dpi = 300 )
 
#%% SH Tropics 
latmask = (glat > -30) & (glat < 0) 
    
fig, ax = plt.subplots(2, 2, figsize = [15, 15])
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")
ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r"0$^{\circ}$ - 30$^{\circ}$S ")
fig.savefig("Figures/jul_sh_trop_iwp.png", bbox_inches = "tight", dpi = 300 )

#%% N.H. Tropics 
latmask = (glat > 0) & (glat < 30) 
    
    
fig, ax = plt.subplots(2, 2, figsize = [15, 15])
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")
ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r"0$^{\circ}$ - 30$^{\circ}$N ")
fig.savefig("Figures/jul_nh_trop_iwp.png", bbox_inches = "tight", dpi = 300 ) 
 
#%% N.H. subTropics 
latmask = (glat > 30) & (glat < 45)     
    
fig, ax = plt.subplots(2, 2, figsize = [15, 15])
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")
ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r"30$^{\circ}N$ - 45$^{\circ}$N ")
fig.savefig("Figures/jul_nh_sub_trop_iwp.png", bbox_inches = "tight", dpi = 300 ) 
#%% N.H. high lats 
latmask = (glat > 45)  
    
fig, ax = plt.subplots(2, 2, figsize = [15, 15])
ax = ax.ravel()

lsmmask = glsm == 0
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[:2])

lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz
plot_pd(landmask, mask, ax[2:4])

for i in range(4):
    ax[i].set_xlim([60, 310])
    ax[i].set_ylim([-5, 50])
    ax[i].set_xlabel("166V GHz [K]")
    ax[i].set_ylabel("PD [K]")
ax[0].set_title("Water (Cloudy)")
ax[1].set_title("Water (Clear)")
ax[2].set_title("Land (Cloudy)")
ax[3].set_title("Land (Clear)")
fig.suptitle(r" > 45$^{\circ}$N ")
fig.savefig("Figures/jul_nh_high_lat_iwp.png", bbox_inches = "tight", dpi = 300 ) 
   

#%% check low IWP over land in cloudy cases
latmask = (glat > 30) #& (glat < 45)  
  
fig, ax = plt.subplots(1, 2, figsize = [15, 7])
ax = ax.ravel()

  
lsmmask = glsm == 1
landmask = np.logical_and(latmask, lsmmask)    
mask = three_sigma(tb[:,:,3][landmask]) # with 183+-3 GHz


lsmmask = glsm == 2
landmask = np.logical_and(latmask, lsmmask)  
tbmask = tb[:, :, 3][landmask] < mask   #  cloudy cases
 
plot_pd(landmask, mask, ax[:2])
   
tbmask = tbmask
tb0     = tb[:, :, 0][landmask][tbmask]
lat0    = glat[landmask][tbmask]
lon0    = glon[landmask][tbmask]
iwp0    = giwp[landmask][tbmask]
    
if tb0.shape[0] > 1000000:
    lat0 = lat0[::50]
    lon0 = lon0[::50]
    iwp0 = iwp0[::50]
    
fig, ax = plt.subplots(1, 1, figsize = [15, 8])
m = Basemap(projection= "cyl", llcrnrlon = 0,  
            llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
m.drawcoastlines()  
cs = m.scatter(lon0, lat0, c = iwp0, s = 2.5, norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
m.drawcountries()
m.drawstates()
parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", shrink = 0.5, ax = ax)
fig.savefig("Figures/jan_IWP_snow_cloudy.png", bbox_inches = "tight", dpi = 300)


