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
from pansat.products.satellite.cloud_sat import l2b_geoprof
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm
from pansat.formats.hdf4 import HDF4File
from scipy import interpolate
import glob
from iwc2tb.common.hist2d import hist2d

import xarray
#%%
class SNOW2C():
    
    def __init__(self, filename):
        
        self.filename = filename
#        file_name = 'MYD06_L2.A2007219.2010.006.2014053202546.hdf'
        self.file = SD(self.filename, SDC.READ)
        
        datasets_dic = self.file.datasets()
        
        print (datasets_dic.keys())
        
        self.SWC        = self.__getitem__("snow_water_content")
        self.R          = self.__getitem__("snowfall_rate")
        self.logN0      = self.__getitem__("log_N0")
        self.loglambda  = self.__getitem__("log_lambda")   
        self.height     = self.__getitem__("Height")
        
        
    def __getitem__(self, key):
        
        sds_obj = self.file.select(key) # select sds
        data = sds_obj.get() 
        return data

#%%
if __name__ == "__main__":        

    path = "/home/inderpreet/Downloads" 
    file = "2006355074548_03451_CS_2B-GEOPROF_GRANULE_P1_R05_E02_F00.hdf"
    file = "2010027071721_19950_CS_2B-GEOPROF_GRANULE_P1_R05_E03_F00.hdf"
    
    
    path1 = "/home/inderpreet/Dendrite/Projects/IWP/GMI/2C-SNOW/"
    file1 = "2006355074548_03451_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf"
    file2 = "2010027071721_19950_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E03_F00.hdf"
    
    
    #file2 = "2010027071721_19950_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E03_F00.hdf"
    
    
    filename = os.path.join(path, file)
    filename1 = os.path.join(path1, file1)   
    filename2 = os.path.join(path, file2)
    
    filenames = glob.glob("/home/inderpreet/Dendrite/SatData/CloudSat/2C-SNOW.R05/*.hdf")
    #dataset = l2b_geoprof.open(filename2)
    
#%%
    SWC    = []
    R      = []
    logN0  = []
    logL   = []
    for filename in filenames[:1]:
        s2c = SNOW2C(filename)        
        SWC.append(s2c.SWC)
        R.append(s2c.R)
        logN0.append(s2c.logN0)
        logL.append(s2c.loglambda) 
        
    SWC   = np.concatenate(SWC, axis = 0)    
    logN0 = np.concatenate(logN0, axis = 0)
    logL  = np.concatenate(logL, axis = 0)
    R     = np.concatenate(R, axis = 0)
    
    SWC     = np.where(SWC < -900, np.nan, SWC)
    logN0   = np.where(logN0 < -900, np.nan, logN0)
    logL    = np.where(logL< -900, np.nan, logL)
    R       = np.where(R< -900, np.nan, R)
    
      
    N0 = 10 ** logN0
    L  = 10 ** logL

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = [20, 8])
    fig.tight_layout(pad=3.0)
    
    ax1.scatter(R, N0, alpha = 0.3)
    ax1.set_yscale("log")    
    ax1.set_xlabel(r"R [mm h$^{-1}$]")
    ax1.set_ylabel(r"N0 [m$^{-3}$ mm$^{-1}$]")
    ax1.text(0.3, 0.8, r'N0 = 3800 R$^{-0.87}$ ', transform=ax1.transAxes)
    
    
    ax2.scatter(R, L, alpha = 0.3)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"R [mm h$^{-1}$]")
    ax2.set_ylabel(r"$\Lambda$ [mm$^{-1}$] ")
    ax2.text(0.3, 0.8, r'$\Lambda$ = 2.55 R$^{-0.48}$ ', transform=ax2.transAxes)

    R1 = np.sort(R.ravel())
    y1 = 3800 * R1 **(-0.87)    
    ax1.plot(R1, y1.ravel(), 'k')
    
    y2 = 2.55 *  R1**(-0.48)
    ax2.plot(R1, y2.ravel(), 'k')
    
    
    ax3.scatter(N0, L, alpha = 0.3)
    ax3.set_yscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"N0 [m$^{-3}$ mm$^{-1}$]]")
    ax3.set_ylabel(r"$\Lambda$ [mm$^{-1}$] ")
    ax3.text(0.3, 0.8, r'$\Lambda$ = 2.55 R$^{-0.48}$ ', transform=ax2.transAxes)
    
    fig.savefig("size_distribution_2CSP.png", bbox_inches = "tight")
    
#%%
    height = s2c.height
    bins   = np.arange(0, 6000, 1)
    Range  = np.arange(0, N0.shape[1], 1)
    XN0 = xarray.DataArray(N0[3000:9000, :], coords = [bins, Range], dims = ['bins', 'range'], name = 'N0')  
    XL  = xarray.DataArray(L[3000:9000, :], coords = [bins, Range], dims = ['bins', 'range'], name = 'lambda') 
    XH  = xarray.DataArray(height[3000:9000, :], coords = [bins, Range], dims = ['bins', 'range'], name = 'height') 
    XR   = xarray.DataArray(R[3000:9000, :], coords = [bins, Range], dims = ['bins', 'range'], name = 'snowfall_rate') 
 
    A = xarray.merge([XN0, XL, XH, XR])
    A.to_netcdf('L2CSNOW.nc', 'w')
    
#%%
    
    
    
    
    
    