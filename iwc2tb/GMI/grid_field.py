#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:20:23 2021

@author: inderpreet
"""
import numpy as np


def grid_field(lat, lon, iwp, gsize = 2.5, startlat = 65.0):
    #startlat = 65.0 
    #gsize = 2.5 # grid size in degree
    nx, ny = int(360/gsize), int(startlat*2/gsize)
    
    field, inds = np.zeros([ny,nx]), np.zeros([ny,nx])  # initialise IWP array
    
    # calculate lat/lon indices for final grid
    ladex = np.round((lat[:]+startlat-gsize*.5) / gsize)
    lodex = np.round((lon[:]-gsize*.5) / gsize)
    
    
    for x in range(nx):
        if np.any(lodex == x):
            dexsub = np.where(lodex == x)[0]
            subset = iwp[dexsub] 
            subsetla = ladex[dexsub]
            for y in range(ny):
                if np.any(subsetla == y):
                    lolasubset = np.where(subsetla == y)[0]
                    field[y,x] += subset[lolasubset].sum() # add up, get mean afterward
                    
                    inds[y,x] += len(subset[lolasubset])
                    
                    
    return field, inds       