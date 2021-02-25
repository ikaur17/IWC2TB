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
plt.rcParams.update({'font.size': 20})

#%%
def bin_iwp(lat, iwp, latbins = None):

    if latbins is None:
        
        latbins  = np.arange(-65, 66, 2)
    
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
    
    for zfile in zipfiles:
        print (zfile)
        dardarfile, N = zip2dardar.zip2dardar(zfile)
        dardar = DARDARProduct(dardarfile, latlims = [-65, 65], node = N)
        
        atm = atmdata(dardar, p_grid, domain = None)
        iwc = np.squeeze(atm.iwc)
        lat = atm.lat
        
        z   = pres2alt(p_grid)

        iwp = np.zeros(lat.shape)

        for i in range(iwc.shape[1]):
            iwp[i] = np.trapz(iwc[:, i], z)
            
        iwp_total.append(iwp)
        lat_all.append(lat)

        
    return np.concatenate(iwp_total), np.concatenate(lat_all)    
#%%
if __name__ == "__main__":    
    # GMI simulations    
    inpath   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.3')  
    inpath1   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_f07') 
 
     
    matfiles = glob.glob(os.path.join(inpath, "2010_*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath1, "2010_*.mat"))    
    
    #matfiles = matfiles = matfiles1
    
    gmi = GMI(matfiles)
    glat = gmi.lat.ravel()
    giwp = gmi.iwp.ravel()


    gmi1 = GMI(matfiles1)
    glat1 = gmi1.lat.ravel()
    giwp1 = gmi1.iwp.ravel()    

#%%    
    
    zipfiles = gmi.get_inputfiles()
    
    diwp, dlat = dardar_iwp(zipfiles)
    
    zipfiles1 = gmi1.get_inputfiles()
    
    diwp1, dlat1 = dardar_iwp(zipfiles1)

#%%    
    iwp_mean_gmi, latbins = bin_iwp(glat, giwp)    
    iwp_mean_dardar, latbins = bin_iwp(dlat, diwp)
    
    iwp_mean_gmi1, latbins = bin_iwp(glat1, giwp1)    
    iwp_mean_dardar1, latbins = bin_iwp(dlat1, diwp1)

#%%    
    fig, ax = plt.subplots(1, 1, figsize = [8, 12])
    ax.plot(iwp_mean_gmi, latbins, 'r--', label = "DARDAR PSD")
    ax.plot(iwp_mean_dardar, latbins, 'r', label = "DARDAR")
    
    ax.plot(iwp_mean_gmi1, latbins, 'b--', label = "f07 PSD")
    ax.plot(iwp_mean_dardar1, latbins,'b', label = "DARDAR")
    
    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("Lat [deg]") 
    # ax.set_xscale('log')
    ax.legend()
    fig.savefig("Figures/IWP_GMI.png", bbox_inches = "tight")    
    
#%%
        
    