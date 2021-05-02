#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 21:12:44 2021

@author: inderpreet
"""


import xarray
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
plt.matplotlib.rcParams.update({'font.size': 16})
import pickle
from era2dardar.RADARLIDAR import DARDAR
from iwc2tb.GMI.grid_field import grid_field  
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 30})
#%%
def get_data(region, lst, lon, lat, y):
    
    latlim = latlims[region]
    lonlim = lonlims[region]
    
    lamask = (lat < latlim[1]) & (lat > latlim[0])
    lomask = (lon < lonlim[1]) & (lon > lonlim[0])
    
    mask = np.logical_and(lamask, lomask)

    nanmask = np.isnan(y)
    
    mask = np.logical_and(~nanmask, mask)
    
    lst = np.array(lst[mask])
    lon = np.array(lon[mask])
    lat = np.array(lat[mask])
    y   = np.array(y[mask])
    
    return lst, lon, lat, y

def append_data(lst, lon, lat, y, LST, LON, LAT, Y):
    LST.append(lst)
    LON.append(lon)
    LAT.append(lat)
    Y.append(y)
    
    return LST, LON, LAT, Y


def get_iwp_mean(LST, Y):

    iwp_mean = []
    for i in range(24):
        lstmask = LST == i
        print(Y[lstmask].shape)
        iwp_mean.append(np.mean(Y[lstmask]))
        
    return iwp_mean  

def get_iwp_mean_gridded(Y):
        
    return np.nanmean(Y, axis = (1, 2))

def bin_iwp(lat, iwp, latbins = None):

    if latbins is None:
        
        latbins  = np.arange(-65, 66, 2)

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins

#%%
regions = ["Africa", "Tropical_Indian_Ocean",
           "North_Pacific_Ocean", "Tropical_Pacific", 
           "South_America", "Maritime_Continent"]


latlims = {"Africa" : [-20, 5],
           "Tropical_Indian_Ocean" : [-20, 0],
           "North_Pacific_Ocean" : [15, 30],
           "Tropical_Pacific" : [-20, 0],
           "South_America" : [-15, 0],
           "Maritime_Continent": [-10, 10]}

lonlims = {"Africa" : [15, 32],
           "Tropical_Indian_Ocean" : [55, 95],
           "North_Pacific_Ocean" : [140, 200],
           "Tropical_Pacific" : [160, 200],
           "South_America" : [280, 320], 
           "Maritime_Continent": [90, 150]}
#%%

path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/")
files = glob.glob(os.path.join(path + "20*IWP_t2m.nc"))

#filename = os.path.join(path, file)


latbins = np.arange(-30, 31, 1.5)


data = {"LON" : [],
            "LAT" : [],
            "LST" : [],
            "Y"   : []}
all_data = {}
for region in regions:
    all_data[region] = {"LON" : [],
                        "LAT" : [],
                        "LST" : [],
                        "Y"   : []} 
bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200])
iwpg = np.zeros([24, 60, 360])
iwpgc = np.zeros([24, 60, 360])
iwpg0 = np.zeros([24, 60, 360])
iwpgc0 = np.zeros([24, 60, 360])
ghist = []
for file in files:
    
    print(file)
    dataset = xarray.open_dataset(file)


    lst = dataset.lst.data
    lon = dataset.lon.data
    lat = dataset.lat.data
    
    y   = dataset.y_mean.data 
    y0  = dataset.y.data
    
    dataset.close()  
    
    
    hist, _ = np.histogram(y, bins)
    
    ghist.append(hist)
    
    nanmask = ~np.isnan(y0)
 
    for ix in range(0, 24):
        
        lstmask = lst == ix
        nanmask = ~np.isnan(y0)
        
        mask = np.logical_and(lstmask, nanmask)
        
        field, inds =  grid_field(lat[mask], lon[mask], y[mask],
                                  gsize = 1.0, startlat = 30.0)
    
        iwpg[ix, :, :]  += field
        iwpgc[ix, :, :] += inds 
        
                
        field1, inds1 =  grid_field(lat[mask], lon[mask], y0[mask],
                                  gsize = 1.0, startlat = 30.0)
        
        iwpg0[ix, :, :]  += field1
        iwpgc0[ix, :, :] += inds1 
    
    nanmask = ~np.isnan(y0)
    ziwp, ziwpc = bin_iwp(lat[nanmask], y[nanmask], latbins)
    ziwp0, ziwp0c = bin_iwp(lat[nanmask], y0[nanmask], latbins)

    outfile = file[:-3] + ".pickle"    
    with open(outfile, "wb") as f:
        pickle.dump(ziwp, f)
        pickle.dump(ziwpc, f)
        pickle.dump(ziwp0, f)
        pickle.dump(ziwp0c, f)

    for ix, region in enumerate(regions):
        print(regions[ix])
        
        lst1, lon1, lat1, y1 = get_data(region, lst, lon, lat, y)
        
        print (lst)
        
        all_data[region]["LON"].append(lon1)
        all_data[region]["LAT"].append(lat1)
        all_data[region]["LST"].append(lst1)
        all_data[region]["Y"].append(y1)

    with open(file[:-3] + "_ghist.pickle", "wb") as f:
        pickle.dump(ghist, f)
        f.close()
    with open(file[:-3] +"gridded.pickle", "wb") as f:
        pickle.dump(iwpg, f)
        pickle.dump(iwpgc, f)
        pickle.dump(iwpg0, f)
        pickle.dump(iwpgc0, f)
        f.close() 

for ix, region in enumerate(regions):
    outfile = region + '.pickle'
    
    with open(outfile, 'wb') as f:

        pickle.dump(all_data[region], f)
        
    f.close()     


#%%
glat = np.arange(-30, 30, 1)   
glon = np.arange(0, 360, 1)
fig, axes = plt.subplots(6, 4, figsize = [60, 30])
fig.subplots_adjust(hspace=-3.5) # height spaces
fig.tight_layout()
#fig.subplots_adjust(wspace=-.1) # width spaces
axes = axes.ravel()
iwpm = iwpg/iwpgc

for ix in range(0, 24):
    
    m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -60, urcrnrlon = 360, urcrnrlat = 60, ax = axes[ix])
    m.drawcoastlines()
    m.pcolormesh(glon, glat, iwpm[ix, :, :], norm=colors.LogNorm(vmin=1e-5, vmax= 15), cmap = cm.rainbow)
    axes[ix].set_title("LST = " + str(ix))
    
    
fig, axes = plt.subplots(6, 4, figsize = [40, 20])
axes = axes.ravel()
for ix in range(0, 24):
    
    m = Basemap(llcrnrlon = 0, llcrnrlat = -30, urcrnrlon = 360, urcrnrlat = 30, ax = axes[ix])
    m.drawcoastlines()
    m.pcolormesh(glon, glat, iwpgc[ix], vmin = 0, vmax = 2000, cmap = cm.rainbow)
    axes[ix].set_title("LST = " + str(ix))    


#%%    
bins = np.arange(0, 24, 1)


for ix, region in enumerate(regions):
    iwpg = []

    outfile = region + '.pickle'
    
    with open(outfile, 'rb') as f:
        all_data = pickle.load(f)

    local_times = np.concatenate(all_data["LST"])
    iwps        = np.concatenate(all_data["Y"])
    lats        = np.concatenate(all_data["LAT"])
    lons        = np.concatenate(all_data["LON"])
    
    for lst in range(0, 24):

        mask = local_times == lst
        field, inds =  grid_field(lats[mask], lons[mask], iwps[mask],
                                  gsize = 2.0, startlat = 30.0)
        
        iwpg.append(field/inds)
        #fig, ax = plt.subplots(1, 1, figsize = [8, 8])
        #cs = ax.pcolormesh(field/inds, vmin = 0, vmax = 4)
        #fig.colorbar(cs, label="IWP [kg/m2]")
        #ax.set_title("local hour =" + str(lst))
        
    iwpg = np.stack(iwpg)    
    fig, ax = plt.subplots(1, 1, figsize = [10, 6])
    mean = get_iwp_mean_gridded(iwpg)

    ax.plot(bins, mean, 'o-', label = region)
    ax.set_title(regions[ix])
    ax.set_xlabel("Local time [h]") 
    ax.set_ylabel("IWP [kg/m2]")  
    ax.grid("on")
#ax.legend()
    fig.savefig(  region + ".png", bbox_inches = "tight")
 

#%%        