#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:25:25 2021

@author: inderpreet
"""


import numpy as np
import xarray
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})




#%%   
def read_nc(file):
    dataset = xarray.open_dataset(file)
    
    
    giwp_mean = dataset.iwp_mean.data
    giwp0 = dataset.iwp0.data
    glon = dataset.lon.data
    glat = dataset.lat.data
    glsm = dataset.lsm.data
    
    return giwp_mean, glon, glat, glsm


#%%
def calculate_iwp_pd(pd, i1, i2, pdbins):
    
    ipd = np.digitize(pd, pdbins )
    
    aro = []
    tro = []
    
    for i in range(pdbins.size -1):
    
        im = ipd == i+1
        
        aro.append(np.mean(i1[im]))
        tro.append(np.mean(i2[im]))
        
    return np.stack(aro), np.stack(tro)     
 #%%       
        

filearo = "jan2020_IWP_lpa1.nc"
filetro = "jan2020_IWP_lpa_pr1.nc"


iwp1, lat, lon ,lsm = read_nc(filearo)
iwp2, _, _ , _      = read_nc(filetro)

nanmask1 = np.isnan(iwp1) | (iwp1 < 0.01)


dataset = xarray.open_dataset("jan2020_tb_lpa.nc")

tb = dataset.tb.data

#%%
pd = tb[:, :,  0] - tb[:, :,  1]

pdbins = np.arange(-5, 40, 2)


# all surface types
ipd = np.digitize(pd[~nanmask1], pdbins )

aro, tro = calculate_iwp_pd(pd[~nanmask1], iwp1[~nanmask1], iwp2[~nanmask1], pdbins)

# water
nanmask = ~nanmask1 & (lsm == 0)
ipd = np.digitize(pd[nanmask], pdbins )

aro1, tro1 = calculate_iwp_pd(pd[nanmask], iwp1[nanmask], iwp2[nanmask], pdbins)

# land
nanmask = ~nanmask1 & (lsm == 1)
ipd = np.digitize(pd[nanmask], pdbins )

aro2, tro2 = calculate_iwp_pd(pd[nanmask], iwp1[nanmask], iwp2[nanmask], pdbins)

# snow
nanmask = ~nanmask1 & (lsm == 2)
ipd = np.digitize(pd[nanmask], pdbins )

aro3, tro3 = calculate_iwp_pd(pd[nanmask], iwp1[nanmask], iwp2[nanmask], pdbins)

# seaice
nanmask = ~nanmask1 & (lsm == 3)
ipd = np.digitize(pd[nanmask], pdbins )

aro4, tro4 = calculate_iwp_pd(pd[nanmask], iwp1[nanmask], iwp2[nanmask], pdbins)


#%%
pdc = (pdbins[1:]  + pdbins[:-1])/2

fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.plot(pdc, aro, 'k', label = "All")
ax.plot(pdc, tro, 'k--')


ax.plot(pdc, aro1, 'r', label = "Water")
ax.plot(pdc, tro1, 'r--')


ax.plot(pdc, aro2, 'g', label = "Land")
ax.plot(pdc, tro2, 'g--')


ax.plot(pdc, aro3, 'm', label = "Snow")
ax.plot(pdc, tro3, 'm--')

ax.plot(pdc, aro4, "b", label = "Seaice")
ax.plot(pdc, tro4, 'b--')

ax.legend()

ax.set_xlabel("Polarisation difference [K]")
ax.set_ylabel(r"IWP [kg m$^{-2}$]")
ax.set_yscale("log")

ax.grid("on", alpha = 0.3)
fig.savefig("IWP_PD_GMI.pdf", bbox_inches = "tight")





















