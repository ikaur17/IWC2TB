#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:11:50 2021

@author: inderpreet
"""
import numpy as np

def filter_gmi_sat(lat, lon, tb, stype,  latlims = None, lsm = None):
        
    mask1 = np.ones(lat.shape, dtype = bool)  
    if latlims is not None:
    
        lat1  = latlims[0]
        lat2  = latlims[1]
        mask1 = (np.abs(lat) >= lat1) & (np.abs(lat) <= lat2)
        
    mask2 = np.ones(lat.shape, dtype = bool)  
    
    if lsm is not None:
         if np.isscalar(lsm):        
             mask2 = stype == lsm             
         else:
             mask2 = np.zeros(lat.shape, dtype = bool)  
             for i in range(len(lsm)):
                 im1 = stype == lsm[i]
                 mask2  = np.logical_or(im1, mask2)  
        
    mask = np.logical_and(mask1, mask2)     
    
    return tb[mask, :], lat[mask], lon[mask]