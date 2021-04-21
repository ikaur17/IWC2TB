#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:10:13 2021

simple class to read in SpareIce data

@author: inderpreet
"""


import xarray
import numpy as np


class spareICE():
    
    def __init__(self, filenames):
        
        
        lon = []
        lat = []
        iwp = []
        for filename in filenames:
            dataset = xarray.open_dataset(filename)
            
            lon.append(dataset.LON.data)
            lat.append(dataset.LAT.data)
            iwp.append(dataset.IWP.data)
        
        
            dataset.close()
            
        self.iwp = np.concatenate(iwp)
        self.lon = np.concatenate(lon)
        self.lat = np.concatenate(lat)
        