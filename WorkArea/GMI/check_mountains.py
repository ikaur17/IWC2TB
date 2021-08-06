#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:11:54 2021

@author: inderpreet
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors

#%%
with open("gridded_iwp.pickle", "rb") as f:
    
    giwpg = pickle.load(f)
    giwpc = pickle.load(f)
    giwpg0 = pickle.load(f)
    giwpc0 = pickle.load(f)
    f.close()
    
#%%
with open("gridded_spareice.pickle", "rb") as f:
    siwpg = pickle.load(f)
    siwpc = pickle.load(f)
    f.close()  

#%%
with open("gridded_dardar.pickle", "rb") as f:
    diwpg = pickle.load(f)
    diwpc = pickle.load(f)
    f.close()    
#%%
   

#%% spatial distribution
lon = np.arange(0, 360, 1)
lat = np.arange(-65, 65, 1)
lon1 = np.arange(0, 360, 2.5)
lat1 = np.arange(-65, 65, 2.5)


fig, axes = plt.subplots(4, 1, figsize = [40, 20])
fig.tight_layout()
ax = axes.ravel()
m = Basemap(projection= "cyl", llcrnrlon = 0,  llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[0])
m.drawcoastlines()  
m.pcolormesh(lon, lat, 0.001 * siwpg/siwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)

parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])


m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[1])
m.pcolormesh(lon1, lat1, diwpg/diwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[2])
cs = m.pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[3])
cs = m.pcolormesh(lon, lat, giwpg0/giwpc0, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
fig.colorbar(cs, label="IWP [kg/m2]", ax = axes)

ax[0].set_title("SpareICE")
ax[1].set_title("DARDAR")
ax[2].set_title("GMI QRNN")
ax[3].set_title("GPROF")
fig.savefig("Figures/IWP_spatial_distribution.png", bbox_inches = "tight")   

#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(1, 1, figsize = [15, 8])
fig.tight_layout()
plt.rcParams.update({'font.size': 20})
lon = np.arange(0, 360, 1)
#lon = (lon + 180) % 360 - 180
m = Basemap(projection= "cyl", llcrnrlon = -180, llcrnrlat = -65, urcrnrlon = 180, urcrnrlat = 65, ax = ax)
cs = m.pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])
m.drawmeridians(meridians,labels=[True,False,False,True])
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="2.5%", pad=0.05)

fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", ax = ax, shrink = 0.55)
ax.set_title("IWP (monthly mean) July 2019")
fig.savefig("IWP_July.png", bbox_inches = "tight", dpi = 300)
#%% plot over himalayas

with open("error_IWP.pickle", "rb") as f:
    iwp   = pickle.load(f)
    tb    = pickle.load(f)
    stype = pickle.load(f)
    lat_e   = pickle.load(f)
    lon_e   = pickle.load(f)
    f.close()
    
#%%
fig, ax = plt.subplots(1, 1, figsize = [12, 6])
fig.tight_layout()

m = Basemap(projection= "cyl", llcrnrlon = 0,  llcrnrlat = -65, 
            urcrnrlon = 360, urcrnrlat = 65)
m.drawcoastlines()  
#im = stype == 3
m.scatter(lon_e[im], lat_e[im], 0.01)

parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])    
    
    

    


 