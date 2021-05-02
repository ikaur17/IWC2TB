#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:51:07 2021

Removes oversampling in GMI observations over high latitudes

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def remove_oversampling_gmi(tb, lat, lon, lsm, iwp):
    """
    remove over-sampling in GMI data with latitudes 

    Parameters
    ----------
    tb : np.array, Brightness temp
    lat : np.array, latitudes
    lon : np.array, longitudes
    lsm : np.array, surface class

    Returns
    -------
    tb_sub : np.array, sampled tb
    lat_sub : np.array
    lon_sub : np.array
    lsm_sub : np.array

    """    

    
    fig, ax = plt.subplots(1, 1, figsize = [20, 10])
    
    bins = np.arange(-65, 66, 1)
    hist = np.histogram(lat.ravel(), bins, density = True) 
    
    ax.hist(lat, bins, density = True)
    
    factors = hist[0]/hist[0].min() 
    
    ilat   = np.digitize(lat, bins)
    icount = np.bincount(ilat)
    
    tb_sub  = []
    lat_sub = []
    lon_sub = []
    lsm_sub = []
    iwp_sub = []
    
    for i in range(1, len(bins)):
        
        # calculate the oversampling factor for each lat bin
        factor = 1 - 1/factors[i-1]
        
        n = np.int(icount[i] * factor) 
 
        iargs = np.where(ilat == i)[0]
        random.shuffle(iargs)  
        iargs_sub = iargs[n:]
        
        tb_sub.append(tb[iargs_sub, :])
        lat_sub.append(lat[iargs_sub])
        lon_sub.append(lon[iargs_sub])
        lsm_sub.append(lsm[iargs_sub])
        iwp_sub.append(iwp[iargs_sub])
        
   
    
    tb_sub = np.vstack(tb_sub)
    lat_sub = np.concatenate(lat_sub)     
    lon_sub = np.concatenate(lon_sub) 
    lsm_sub = np.concatenate(lsm_sub)
    iwp_sub = np.concatenate(iwp_sub)


    ax.hist(lat_sub, bins, density = True)   
    ax.set_ylabel("number density")
    ax.set_xlabel("Latitude")
    ax.legend(["original", "sampled"])    

        
        
        
    return tb_sub, lat_sub, lon_sub, lsm_sub, iwp_sub   



       
        