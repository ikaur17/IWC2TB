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

  

path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/")
files = glob.glob(os.path.join(path + "*_IWP.nc"))


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
iwpg = np.zeros(iwpg.shape[0])
for file in files:
    outfile = file[:-3] + "_regridded.pickle"
    with open(outfile, 'wb') as f:
    
        iwpg   += pickle.load(f)
        counts += pickle.load(f)
   
    f.close()
    
    
    


lats = np.arange(-30, 30, 2.5)
lons = np.arange(0, 360, 2.5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [20, 20])
   
cs = ax1.pcolormesh(lons, lats, iwpg/counts, vmin = 0, vmax = 25, cmap = cm.rainbow)
ax1.set_title("retrieved WVP")


ax2.set_title("ERA5")
fig.colorbar(cs, label="IWP [kg/m2]", ax = [ax1, ax2])
ax1.set_ylabel("latitude [deg]")
ax2.set_ylabel("latitude [deg]")
ax2.set_xlabel("longitude [deg]")

fig.savefig("gridded_IWP.png", bbox_inches = "tight")






















        
        
            
        
        