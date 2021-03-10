#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:13:02 2021

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from iwc2tb.GMI.GMI import GMI
import zipfile
import typhon.arts.xml as xml
from era2dardar import zip2dardar
from era2dardar.DARDAR import DARDARProduct
from era2dardar.atmData import atmdata
from era2dardar.utils.alt2pressure import alt2pres, pres2alt
from iwc2tb.common.plot_locations_map import plot_locations_map
import shutil
plt.rcParams.update({'font.size': 20})

#%%
def bin_iwp(lat, iwp, latbins = None):

    if latbins is None:
        
        latbins  = np.arange(-65, 66, 2.5)
    
    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)/nbins
    
    return iwp_mean, latbins

#%%    
def dardar_iwp(zipfiles):
    
    p_grid = alt2pres(np.arange(-700, 20000, 250))
    p_grid = (np.concatenate([p_grid, 
                             np.array([30, 20, 10, 7, 5, 3, 2, 1]) * 100]))
    
    iwp_total = []
    lat_all = []
    lon_all = []
    
    for zfile in zipfiles:
        print (zfile)
        dardarfile, N = zip2dardar.zip2dardar(zfile)
        dardar = DARDARProduct(dardarfile, latlims = [-65, 65], node = N)
        
        atm = atmdata(dardar, p_grid, domain = None)
        iwc = np.squeeze(atm.iwc)
        lat = atm.lat
        lon = atm.lon
        
        z   = pres2alt(p_grid)

        iwp = np.zeros(lat.shape)

        for i in range(iwc.shape[1]):
            iwp[i] = np.trapz(iwc[:, i], z)
            
        iwp_total.append(iwp)
        lat_all.append(lat)
        lon_all.append(lon)
        
    return np.concatenate(iwp_total), np.concatenate(lat_all), np.concatenate(lon_all)     

#%%
def filter_iwp(latlims, lonlims, lat, lon):
    
    im1  = (lat > latlims[0]) & (lat < latlims[1])
    im2  = (lon > lonlims[0]) & (lon < lonlims[1])
    mask = np.logical_and(im1, im2)
    return mask

#%%
def get_basenames(files):
    basefiles = []
    for file in files:   
        basefiles.append(os.path.basename(file))
    return basefiles    

def get_fullnames(path , files):
    
    fullnames = []    
    for file in files:
        fullnames.append(os.path.join(path, file))        
    return fullnames    
#%%
if __name__ == "__main__":    
    # GMI simulations    
    inpath   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.3')  
    inpath1  =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_f07')
    inpath2  =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_si')

    
    
    matfiles = glob.glob(os.path.join(inpath, "2010_*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath1, "2010_0*.mat")) 
    matfiles2 = glob.glob(os.path.join(inpath2, "2010_0*.mat")) 


#%% find files over himalayas with high IWC
    for file in matfiles1:
            gmi = GMI(file)
            
            mla = (gmi.lat > 29) & (gmi.lat < 44)
            mlo = (gmi.lon > 71) & (gmi.lon < 76)
            
            mask = np.logical_and(mla, mlo)
            
            if np.sum(mask) != 0:
                print (file)
                plot_locations_map(gmi.lat, gmi.lon, gmi.iwp)
        


#%%    
    # matfiles1 = matfiles1 + matfiles2

    basefiles   = get_basenames(matfiles)
    basefiles1  = get_basenames(matfiles1)
    basefiles2  = get_basenames(matfiles2)
    commonfiles = set(basefiles).intersection(basefiles1)
    
    
    matfiles    = get_fullnames(inpath, commonfiles)
    matfiles1   = get_fullnames(inpath1, commonfiles)
    matfiles2   = get_fullnames(inpath2, commonfiles)
    
    

    
    gmi = GMI(matfiles)
    glat = gmi.lat.ravel()
    glon = gmi.lon.ravel()%360
    giwp = gmi.iwp.ravel()


    gmi1 = GMI(matfiles1)
    glat1 = gmi1.lat.ravel()
    glon1 = gmi1.lon.ravel()%360
    giwp1 = gmi1.iwp.ravel()    

#%%

  
#%%    
    
    zipfiles = gmi.get_inputfiles()
    
    diwp, dlat, dlon = dardar_iwp(zipfiles)    
    dlon = dlon%360
    
    zipfiles1 = gmi1.get_inputfiles()
    
    diwp1, dlat1, dlon1 = dardar_iwp(zipfiles1)
    dlon1 = dlon1%360
    
    stype  = gmi.stype
    stype1 = gmi1.stype
    
#%% 
    # himalaya   
    latlims = [30, 45]
    lonlims = [60, 105]
    
    #latlims = [0, 0.1]
    #lonlims = [1, 1.1]
    
    dim1 = filter_iwp(latlims, lonlims, dlat1, dlon1)
    dim  = filter_iwp(latlims, lonlims, dlat, dlon)
    
    im   = filter_iwp(latlims, lonlims, gmi.lat, gmi.lon)
    im1  = filter_iwp(latlims, lonlims, gmi1.lat, gmi1.lon)
    
    
    # # N America
    # latlims = [34, 40]
    # lonlims = [240, 242]
    
    
    # dim12 = filter_iwp(latlims, lonlims, dlat1, dlon1)
    # dim2  = filter_iwp(latlims, lonlims, dlat, dlon)
    
    # im2   = filter_iwp(latlims, lonlims, gmi.lat, gmi.lon)
    # im12  = filter_iwp(latlims, lonlims, gmi1.lat, gmi1.lon)
    

    # im   = np.logical_or(im, im2)
    # im1  = np.logical_or(im1, im12)
    # dim1 = np.logical_or(dim1, dim12)
    # dim  = np.logical_or(dim, dim2)
    
#%%
    iwp_mean_gmi, latbins = bin_iwp(glat[~im], giwp[~im])    
    iwp_mean_dardar, latbins = bin_iwp(dlat[~dim], diwp[~dim])
    
    iwp_mean_gmi2 , latbins = bin_iwp(glat1[~im1], giwp1[~im1])    
    iwp_mean_dardar1, latbins = bin_iwp(dlat1[~dim1], diwp1[~dim1])
    
    iwp_mean_gmi1, latbins =  bin_iwp(glat1, giwp1)  
    

#%%    
    fig, ax = plt.subplots(1, 1, figsize = [8, 12])
    ax.plot(iwp_mean_gmi, latbins, 'r--', label = "DARDAR PSD")
    ax.plot(iwp_mean_dardar, latbins, 'r', label = "DARDAR")
    
    ax.plot(iwp_mean_gmi1, latbins, 'b--', label = "f07 PSD")
    #ax.plot(iwp_mean_gmi2, latbins, 'r--', label = "f07 PSD filtered")
    ax.plot(iwp_mean_dardar1, latbins,'b', label = "DARDAR")
    
    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("Lat [deg]") 
    # ax.set_xscale('log')
    ax.legend()
    fig.savefig("Figures/IWP_GMI_dardar.png", bbox_inches = "tight")    
    
#%%
    maplims = [25., 47., 60., 107.]
    plot_locations_map(glat1[im1], glon1[im1], z = giwp1[im1], maplims = maplims)
    plot_locations_map(dlat1[dim1], dlon1[dim1], z = diwp1[dim1], maplims = maplims)    
    