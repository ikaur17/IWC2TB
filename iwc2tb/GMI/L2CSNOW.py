#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:08:27 2021

@author: inderpreet
"""
from pyhdf.SD import SD, SDC
from pyhdf.VS import VS
from pyhdf.HDF import HDF
import os
import numpy as np
import matplotlib.pyplot as plt
from era2dardar.utils.Z2dbZ import Z2dbZ
from pansat.products.satellite.cloud_sat import l2b_geoprof, l2c_snow
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm
from pansat.formats.hdf4 import HDF4File
from scipy import interpolate

class SNOW2C():
    
    def __init__(self, filename):
        
        self.filename = filename
#        file_name = 'MYD06_L2.A2007219.2010.006.2014053202546.hdf'
        self.file = SD(self.filename, SDC.READ)
        
        datasets_dic = self.file.datasets()
        
        print (datasets_dic.keys() )

if __name__ == "__main__":        

    path = "/home/inderpreet/Downloads" 
    file = "2006355074548_03451_CS_2B-GEOPROF_GRANULE_P1_R05_E02_F00.hdf"
    file = "2010027071721_19950_CS_2B-GEOPROF_GRANULE_P1_R05_E03_F00.hdf"
    
    
    path1 = "/home/inderpreet/Dendrite/Projects/IWP/GMI/2C-SNOW/"
    file1 = "2006355074548_03451_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf"
    file1 = "2010027071721_19950_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E03_F00.hdf"
    
    
    file2 = "2010027071721_19950_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E03_F00.hdf"
    
    
    filename = os.path.join(path, file)
    filename1 = os.path.join(path1, file1)   
    filename2 = os.path.join(path, file2)
    
    s2c = SNOW2C(filename1)
    sds_obj = s2c.file.select("snow_water_content") # select sds
    swc = sds_obj.get() 
    
    dataset = l2b_geoprof.open(filename)
    
    start = 10000
    end = 25000
    stride = 100
    z = dataset["radar_reflectivity"]/100
    z = np.where(z < -30, np.nan, z)
    lats = dataset["latitude"]
    bins = dataset["bins"]
    lons = dataset["longitude"]
    height = dataset["height"]
    
    #%% interpolate to common height
    
    h   = np.arange(0, 10000, 250)  
    grid_z = np.zeros([lats.shape[0], h.shape[0]])
    for i in range(lats.shape[0]): 
        f               = (interpolate.interp1d(height[i, :], z[i, :],
                                           fill_value = "extrapolate"))
        grid_z[i, :]          = f(h) 
    
    #%%
    
    
    
    mask = (lats >= 35.5) & (lats <= 36)
    mask1  = (lons < 100) & (lons > 60)
    mask = np.logical_and(mask, mask1)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    cs = ax.pcolormesh(lats[mask], h/1000, grid_z[mask, :].T, norm=Normalize(-30, 40), cmap = cm.rainbow)
    fig.colorbar(cs, label="dbZe")
    ax.set (ylim=(0, 10))
    #plt.gca().invert_yaxis()
    plt.xlabel(r"Latitude [deg]")
    plt.ylabel("Height [km]");
    plt.title("L2B-GEOPROF", loc="left")
    fig.savefig("cloudsat_radar_reflectivity.pdf", bbox_inches = "tight")
    
    swc1 = np.where(swc < 0, np.nan, swc)
    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(lats[mask], height[1, :]/1000, swc1[mask, :].T, norm=Normalize(0, 0.5), cmap = cm.rainbow)
    # plt.colorbar(label="Snow water content [g/m3] ")
    # #plt.gca().invert_yaxis()
    # plt.xlabel(r"Latitude [$^\circ\ N$]")
    # ax.set (ylim=(0, 10))
    # plt.ylabel("Height[km]");
    # plt.title("L2C-SNOW", loc="left")
    
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    cs = ax.pcolormesh(lats[mask], height[1, :]/1000, swc1[mask, :].T /1000, norm=Normalize(0, 0.002), cmap = cm.rainbow)
    fig.colorbar(cs, label="SWC [kg/m3]")
    ax.set (ylim=(0, 10))
    #plt.gca().invert_yaxis()
    plt.xlabel(r"Latitude [deg]")
    plt.ylabel("Height [km]");
    plt.title("L2C-SNOW", loc="left")
    #fig.savefig("cloudsat_radar_reflectivity.pdf", bbox_inches = "tight"
