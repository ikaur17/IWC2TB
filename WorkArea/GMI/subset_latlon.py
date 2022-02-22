#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:12:43 2022

@author: inderpreet
"""
import os
import numpy as np
import xarray
import matplotlib.pyplot as plt




def call_rect(lat, lon):
    

    latmin = round(lat.min()*2)/2
    latmax = round(lat.max()*2)/2
    
    lonmin = round(lon.min()*2)/2
    lonmax = round(lon.max()*2)/2
    
    
    
    
    
    stepsize = 0.5
    newlon, newlat = np.meshgrid(np.arange(lonmin-0.5, lonmax+ 0.5, stepsize), 
                                 np.arange(latmin-0.5, latmax+0.5, stepsize))
    
    t = np.zeros(newlon.shape)
    t[:, :] = 1
    
    
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            
            ilon = round(lon[i, j] * 2) /2
            ilat = round(lat[i, j] * 2) /2
            
            arg1 = np.argwhere((newlon == ilon) & (newlat == ilat))[0]
            
            t[arg1[0], arg1[1]] = 0
            
    t = get_rect(t)
    
    lo1, la1, lo2, la2  = newlon[t[0], t[1]], newlat[t[0], t[1]], newlon[t[2], t[3]], newlat[t[2],t[3]]

    return lo1, lo2, la1, la2       
        

def get_rect(a):
    nrows = a.shape[0]
    ncols = a.shape[1]
    skip = 1
    area_max = (0, [])

    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(nrows):
        for c in range(ncols):
            if a[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])
    
    #print('area', area_max[0])
    #for t in area_max[1]:
    #    print('Cell 1:({}, {}) and Cell 2:({}, {})'.format(*t))
        
    return area_max[1]   

    

file = os.path.expanduser("~/Downloads/adaptor.mars.external-1643356686.3328688-1909-1-57441a8d-3f61-40e8-b9bf-2a7cbbf1fd7d.nc")

dataset = xarray.open_dataset(file)

lat = dataset.latitude.data
lon = dataset.longitude.data
lon = (lon + 180) % 360 - 180

temp = dataset.t[0, :, :].data.ravel()

lo1, lo2, la1, la2 = call_rect(lat, lon)

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.scatter(lon, lat, c = "tab:blue")

ax.plot([lo1, lo1, lo2, lo2, lo1, lo2, lo1, lo2], [la1,la2, la1, la2, la1, la1, la2, la2])

points = np.concatenate([lon, lat], axis = 1)

stepsize = 0.25
newlon, newlat = np.meshgrid(np.arange(lo1, lo2, stepsize), 
                             np.arange(la1, la2, stepsize))
newlon = newlon.ravel()
newlat = newlat.ravel()

newpoints  = np.concatenate([newlon.reshape(-1, 1), 
                             newlat.reshape(-1, 1)], axis = 1)



newt = delanuay_interp(points.reshape(-1, 2), temp, newpoints)


fig, ax = plt.subplots(2, 1, figsize = [10, 20])
ax = ax.ravel()
ax[0].scatter(lon, lat, c= temp, vmin = 220, vmax = 270)
ax[0].plot([lo1, lo1, lo2, lo2, lo1, lo2, lo1, lo2], [la1,la2, la1, la2, la1, la1, la2, la2])
#ax[0].set_xlim([, 100])
ax[1].scatter(newlon, newlat, c = newt, vmin =220, vmax = 270)
for i in range(2):
    ax[i].set_ylim([60, 85])
    ax[i].set_xlim([-20, 80])



