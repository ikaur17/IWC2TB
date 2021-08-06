#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:41:42 2021

@author: inderpreet
"""


from iwc2tb.GMI.GMI_SatData import GMI_Sat
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import pickle
import xarray
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


def check_latlonlims(lat, lon, lst, iwp, latlims, lonlims, counts, iwp_all):
    lamask = np.logical_and(lat > latlims[0] , lat < latlims[1])
    lomask = np.logical_and(lon > lonlims[0] , lon < lonlims[1])    

    mask = np.logical_and(lamask, lomask)
    
    if np.sum(mask) > 0:
        
        ilst = int(lst[mask].min().strftime("%H"))
        idoy = int(lst[mask].min().strftime("%j"))
        
        counts[ilst, idoy] = np.sum(mask)
        iwp_all[ilst, idoy] = np.sum(iwp[mask])
    return counts, iwp_all

def get_pixtime(sctime):
        

        year    = sctime["Year"].data
        mon     = sctime["Month"].data
        day     = sctime["DayOfMonth"].data.astype('timedelta64[D]').astype(np.int32)
        hour    = sctime["Hour"].data.astype('timedelta64[h]').astype(np.int32)
        minute  = sctime["Minute"].data.astype('timedelta64[m]').astype(np.int32)
        sec     = sctime["Second"].data.astype('timedelta64[s]').astype(np.int32)
        
        date    = [datetime(year[i], mon[i], day[i], hour[i], minute[i], sec[i]) for i in range(len(year))]
        
        return np.array(date)
        
def get_lst(time):
        
        t    = time.reshape(-1, 1)
        t  = np.tile(t, lon.shape[1])
        mins = lon * 4.0
        
        lst = t.copy()
        nx = lon.shape[0]
        ny = lat.shape[1]
        lst = [[t[i, j] + timedelta(minutes = np.float(mins[i, j])) for i in range(nx)] for j in range(ny)]

        
        return np.stack(lst).T
    
    
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
months = ["01"]
counts1 = np.zeros([24, 366])
counts2 = np.zeros([24, 366])

iwp1 = np.zeros([24, 366])
iwp2 = np.zeros([24, 366])

for month in months:
    print ("doing month = ", month)
    gpath = os.path.expanduser("~/Dendrite/SatData/GMI/L1B/2019/")
    gfiles = glob.glob(os.path.join(gpath, month, "*", "*HDF5"))
    
    latlims1 = [1., 4.5]
    lonlims1 = [5.0001, 8.5] 
    
    latlims2 = [-3.55, 0.]
    lonlims2 = [24.92, 29] 
    
    for ix, gfile in enumerate(gfiles):
        
        try:
            gmi      = GMI_Sat(gfile)
        except:
            print("GPROF not available for" , gfile)
            continue
        lst      = gmi.lst
        iwp      = gmi.iwp
        lat      = gmi.lat
        lon      = gmi.lon
        
        

        counts1, iwp1 = check_latlonlims(lat, lon, lst, iwp, latlims1, lonlims1, 
                                    counts1, iwp1)
        counts2, iwp2 = check_latlonlims(lat, lon, lst, iwp, latlims2, lonlims2, 
                                    counts2, iwp2)        
        
        
with open("lst_dot_4-12.pickle", "wb")as f:
    pickle.dump(iwp1, f)
    pickle.dump(counts1, f)
    pickle.dump(iwp2, f)
    pickle.dump(counts2, f)
    f.close()        
  
#%%
        
with open("lst_dot_4-12.pickle", "rb")as f:
    iwp1 = pickle.load( f)
    counts1 = pickle.load(f)
    iwp2 = pickle.load( f)
    counts2 = pickle.load( f)
    f.close()        
      
#%%s      
counts1[counts1 == 0] = np.nan
counts2[counts2 == 0] = np.nan

    
fig, ax = plt.subplots(2, 1, figsize = [10, 20])
ax      = ax.ravel()  


ax[0].pcolormesh(counts1[:, :])
ax[1].pcolormesh(iwp1[:, :]/counts1, vmin = 1e-2, vmax = 2.5, cmap = cm.rainbow)
ax[0].grid("on")   
ax[1].grid("on")    
    
#%%

lst_iwp1 = np.sum(iwp1, axis = 1)
lst_cou1 = np.sum(counts1, axis = 1)

lst_iwp2 = np.sum(iwp2, axis = 1)
lst_cou2 = np.sum(counts2, axis = 1)

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot(lst_iwp1/lst_cou1, label = "ocean")
ax.plot(lst_iwp2/lst_cou2, label = "land")
ax.set_xlabel("Local Time [hrs]")
ax.set_ylabel(r"IWP [kg m$^{-2}$]")
ax.legend()
fig.savefig("Figures/diurnal_cycle_ocean_land.png", bbox_inches = "tight", dpi = 300)


landfile = os.path.join("/home/inderpreet/Dendrite/UserAreas/Adria/public/IWP_predictions/CompareGMI/results/CNN", 
                       "land_iwp.nc")

land = xarray.open_dataset(landfile)









