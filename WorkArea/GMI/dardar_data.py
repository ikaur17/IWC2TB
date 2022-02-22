#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:52:34 2021

@author: inderpreet
"""

import glob
import os
import numpy as np
from era2dardar.RADARLIDAR import DARDAR
import pickle


#----------------------------------------------
outfile = "dardar_jan2019"
year = "2019"
month = "01"
#----------------------------------------------

dfiles = []
#dpath  = "/home/inderpreet/Dendrite/SatData/DARDAR/2008/12/"
#dfiles += glob.glob(os.path.join(dpath + "*/*.hdf"))  


dpath  = os.path.join("/home/inderpreet/Dendrite/SatData/DARDAR/", year, month)
#dfiles += glob.glob(os.path.join(dpath , "*/*.hdf"))  

dfiles += glob.glob(os.path.join(dpath , "*/*.nc"))  


#dpath  = "/home/inderpreet/Dendrite/SatData/DARDAR/2009/02/"
#dfiles += glob.glob(os.path.join(dpath + "*/*.hdf"))  

DIWP = []
DLAT = []
DLON = []
for dfile in dfiles[:]:
    dardar =  DARDAR(dfile)
    print (dfile)
    
    diwc    = dardar.iwc
    dlat    = dardar.latitude
    dlon    = dardar.longitude
    dh      = dardar.height

    #lamask = (dlat > 25) & (dlat < 45)
    #lomask = (dlon > 70) & (dlon < 95)
    #dmask = np.logical_and(lamask, lomask)
    
    dmask   = np.abs(dlat) <= 65.0
    
    
    diwp = np.sum(diwc[dmask], axis = 1) * 60
    #diwp    = np.zeros(dlat[dmask].shape[0])    
    #for i in range(diwc[dmask, :].shape[0]):
    #    diwp[i] = np.trapz(diwc[i, :], dh)
    

    DIWP.append(diwp)
    DLAT.append(dlat[dmask])
    DLON.append(dlon[dmask])

diwp = np.concatenate(DIWP)
dlat = np.concatenate(DLAT)
dlon = np.concatenate(DLON)



with open(outfile + ".pickle", "wb") as f:
    pickle.dump(dlat, f)
    pickle.dump(dlon, f)
    pickle.dump(diwp, f)
f.close()


#%%
with open(outfile + ".pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors

fig, ax = plt.subplots(1, 1, figsize = [15, 8])
m = Basemap(projection= "cyl", llcrnrlon = 60,  
            llcrnrlat = 20, urcrnrlon = 105, urcrnrlat = 50, ax = ax)
m.drawcoastlines()  
cs = m.scatter(dlon, dlat, c = diwp, s = 2.5, 
               norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
m.drawcountries()
m.drawstates()
parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", shrink = 0.5, ax = ax)
fig.savefig("Figures/dardar_actual.png", bbox_inches = "tight", dpi = 300)

#%%
import xarray
train = xarray.open_dataset("/home/inderpreet/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train.nc") 

tiwp = train.ta.iwp
tlat = train.ta.lat
tlon = train.ta.lon% 360

lamask = (tlat > 25) & (tlat < 45)
lomask = (tlon > 70) & (tlon < 95)
tmask = np.logical_and(lamask, lomask)



fig, ax = plt.subplots(1, 1, figsize = [15, 8])
m = Basemap(projection= "cyl", llcrnrlon = 60,  
            llcrnrlat = 20, urcrnrlon = 105, urcrnrlat = 50, ax = ax)
m.drawcoastlines()  
cs = m.scatter(tlon[tmask], tlat[tmask], c = tiwp[tmask], s = 2.5, 
               norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
m.drawcountries()
m.drawstates()
parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", shrink = 0.5, ax = ax)
fig.savefig("Figures/dardar_training.png", bbox_inches = "tight", dpi = 300)



