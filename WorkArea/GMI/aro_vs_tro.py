#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:50:46 2021

@author: inderpreet
"""


import matplotlib.pyplot as plt
import numpy as np
import xarray
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap

def zonal_mean(lat, iwp, latbins):
    

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins, )
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins

#%% read data

aro = xarray.open_dataset("jan2020_IWP_aro2.nc")
tro = xarray.open_dataset("jan2020_IWP_tro2.nc")

#%% plot scatter 1-1
bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,
                 .25,.5,1,2, 5,7,8, 9,10,12,13, 14, 15, 16, 17, 18, 19, 20])

bin_center = (bins[1:] + bins[:-1])/2

mask = aro.lsm.data == 0

ahist, _ = np.histogram(aro.iwp_mean.data.ravel()[mask.ravel()], bins, density = True)
thist, _ = np.histogram(tro.iwp_mean.data.ravel()[mask.ravel()], bins, density = True)


fig, ax = plt.subplots(5, 1, figsize = [5, 40])
#fig.tight_layout()
ax = ax.ravel()



ax[0].plot(bin_center, ahist,  label = "ARO")
ax[0].plot(bin_center, thist,   label = "TRO")
ax[0].set_yscale("log")
ax[0].set_xscale("log")
ax[0].legend()
ax[0].set_xlabel("IWP [kg/m2]")
ax[0].set_ylabel("PDF")


for itype, i in zip(["water", "land", "snow", "seaice"],[0, 1, 2, 3]):

    lsmmask = tro.lsm.data == i
    ax[1+i].scatter(aro.iwp_mean.data[lsmmask][::50], tro.iwp_mean.data[lsmmask][::50])
    ax[1+i].set_xlabel("IWP ARO [kg/m2]")
    ax[1+i].set_ylabel("IWP TRO [kg/m2]")
    ax[1+i].set_xlim([1e-4, 20])
    ax[1+i].set_ylim([1e-4, 20])
    x = np.arange(0, 35, 1)
    y = x
    ax[1+i].plot(x, y, 'k')
    ax[1+i].text(7.5, 15, itype)
    #ax[1+i].set_xscale("log")
    #ax[1+i].set_yscale("log")
fig.savefig("Figures/ARO_TRO_scatter.png", bbox_inches = "tight",)

#%%
tbfile = "jan2020_tb_aro2.nc"
dataset2 = xarray.open_dataset(tbfile)

#%% plot zonal means

latbins = np.arange(-65, 66, 1)

aiwp = aro.iwp_mean.data.ravel()
tiwp = tro.iwp_mean.data.ravel()
lat = tro.lat.data.ravel()
lon = tro.lon.data.ravel()

nanmask = ~np.isnan(aiwp)
mask    = aro.lsm.data.ravel() == 0
pd  = (dataset2.tb[:, :, 0] - dataset2.tb[:, :, 1]).data.ravel()
pdmask = (pd > 10) & (pd < 20)
tbmask  = (dataset2.tb.data[:,: , -1] < 250  ).ravel() 

tbmask = np.logical_and(pdmask, tbmask)
mask = np.logical_and(tbmask, mask)
    
nanmask = np.logical_and(nanmask, mask)
aiwp0, aiwp0c = zonal_mean(lat[nanmask], aiwp[nanmask], latbins)

nanmask = ~np.isnan(tiwp)
mask    = aro.lsm.data.ravel() == 0

pd  = (dataset2.tb[:, :, 0] - dataset2.tb[:, :, 1]).data.ravel() 
pdmask = (pd > 10) & (pd < 20)
tbmask  = (dataset2.tb.data[:,: , -1] < 250  ).ravel() 

tbmask = np.logical_and(pdmask, tbmask)
mask = np.logical_and(tbmask, mask)

nanmask = np.logical_and(nanmask, mask)


tiwp0, tiwp0c = zonal_mean(lat[nanmask], tiwp[nanmask], latbins)

fig, ax = plt.subplots(1, 1, figsize = [15, 15])
ax.plot(aiwp0[:-1]/aiwp0c[:-1], latbins, 'b-',  label = "ARO") 
ax.plot(tiwp0[:-1]/tiwp0c[:-1], latbins, 'r--',  label = "TRO") 


ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.legend()
fig.savefig("Figures/zonal_mean_aro_tro.png", bbox_inches = "tight")

#%% plot PDs



mask = tro.lsm.data == 0 
#mask = np.logical_and(mask1, mask2)
#mask = np.logical_and(mask, mask3)
aiwp = aro.iwp_mean.data
tiwp = tro.iwp_mean.data
tb = dataset2.tb.data
fig, ax = plt.subplots(2, 1, figsize = [8, 16])
ax = ax.ravel()
pd = (tb[:, :, 0] - tb[:, :, 1])
cs = ax[0].scatter(tb[:, :, -1][mask][::250], pd[mask][::250], 
                   c = aiwp[mask][::250], norm=colors.LogNorm(vmin=1e-4, vmax= 20), 
                  cmap = cm.gist_ncar)    
ax[0].set_xlabel("TB 166V GHz [K]")
ax[0].set_ylabel("PD 166V - 166H [K]")
ax[0].set_xlim([50, 310])
ax[0].set_ylim([-10, 60])
ax[0].set_title("ARO")
cs = ax[1].scatter(tb[:, :, -1][mask][::250], pd[mask][::250], 
                   c = tiwp[mask][::250], norm=colors.LogNorm(vmin=1e-4, vmax= 20), 
                  cmap = cm.gist_ncar) 
  
ax[1].set_xlabel("TB 166V GHz [K]")
ax[1].set_ylabel("PD 166V - 166H [K]")
ax[1].set_xlim([50, 310])
ax[1].set_ylim([-10, 60])
ax[1].set_title("TRO")
fig.colorbar(cs, ax = ax)
fig.savefig("Figures/PD_water.png", bbox_inches = "tight", dpi = 300)

#%%

fig, ax = plt.subplots(2, 1, figsize = [20, 10])
ax = ax.ravel()
mask = tro.lsm.data == 0
parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, 
            urcrnrlat = 65, ax = ax[0])
m.drawcoastlines()   
mask = mask.ravel()
cs = m.scatter(lon[mask][::50], lat[mask][::50],  c =  tiwp[mask][::50], 
               norm=colors.LogNorm(vmin=1e-4, vmax= 7.5),  cmap = cm.rainbow)
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
m.drawcoastlines() 
ax[0].set_title("TRO")

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, 
            urcrnrlat = 65, ax = ax[1])


cs = m.scatter(lon[mask][::50], lat[mask][::50],  c =  aiwp[mask][::50], 
               norm=colors.LogNorm(vmin=1e-4, vmax= 7.5),  cmap = cm.rainbow)

ax[1].set_title("ARO")

# labels = [left,right,top,bottom]
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

fig.colorbar(cs, ax = ax)

#%%
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.scatter(a.ta[:, 0] - a.ta[:, 1], a.ta[:, -1], c = a.ta.iwp, 
           norm=colors.LogNorm(vmin=1e-4, vmax= 20.5),  cmap = cm.rainbow)

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.scatter(b.ta[:, 0] - b.ta[:, 1], b.ta[:, -1], c = b.ta.iwp, 
           norm=colors.LogNorm(vmin=1e-4, vmax= 20.5),  cmap = cm.rainbow)
#%%
# pd = ta.data[:, 0] - ta.data[:, 1]
# tb = ta.data[:, 0]
# pr = ta.pratio
# stype = ta.stype
# mask = stype == 3
# mask1 = pr < 1.02
# mask = np.logical_and(mask, mask1)
# fig, ax = plt.subplots(1, 1, figsize = [8, 8])
# cs  = ax.scatter(tb[mask], pd[mask], c = pr[mask], vmin = 1, vmax = 1.4, cmap = cm.tab20)
# fig.colorbar(cs)