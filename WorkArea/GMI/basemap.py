#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:48:04 2021

@author: inderpreet
"""

import cmocean
import xarray
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
import pickle
plt.rcParams.update({'font.size': 20})
import time



with open("high_iwp_files_log.pickle", "rb") as f:
    h_iwp_files = pickle.load(f)    
    f.close()

fig, ax = plt.subplots(1, 1, figsize = [20, 12])
m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -90, urcrnrlon = 360, urcrnrlat = 90, ax = ax)
m.drawcoastlines()  

for file in h_iwp_files:

    print (file)
    dataset = xarray.open_dataset(file)
    
 
    mask = dataset.iwp_mean > 20.0
    
    cs = m.scatter(dataset.lon.data[mask.data], 
                   dataset.lat.data[mask.data],
                   c =  dataset.iwp.data[mask.data], vmin = 20, vmax = 50,  cmap = cm.rainbow)
fig.colorbar(cs, ax = ax, shrink = 0.7)
parallels = np.arange(-80.,80,20.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(0.,360.,40.)
m.drawmeridians(meridians,labels=[True,False,False,True])
ax.set_title("QRNN retrieval IWP > 20 kg/m2, January 2020")
fig.savefig("anomalous_IWP_log.png", bbox_inches= "tight")



