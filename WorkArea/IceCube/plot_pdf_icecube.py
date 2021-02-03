#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:05:09 2021

testing IceCube files for errors
also plots the PDF of antenna weighted TB

@author: inderpreet
"""


from typhon.arts.xml import load
import numpy as np
import matplotlib.pyplot as plt
import os
#import zipfile
import shutil
from iwc2tb.IceCube.IceCube import IceCube
import glob
from iwc2tb.py_atmlab.gaussfilter import gaussfilter
import scipy.io
import zipfile
import typhon.arts.xml as xml
from era2dardar.utils.alt2pressure import pres2alt
plt.rcParams.update({'font.size': 22})

def compare_scatter_iwp(matfile):
    """
    compare the iwp from DARDAR and ARTS simulations for one file

    Parameters
    ----------
    matfile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print (matfile)
    zfile = os.path.join(dardarpath , os.path.basename(matfile))
    zfile = zfile.replace(".mat", ".zip")
    
    with zipfile.ZipFile(zfile, 'r') as zip_ref:
        zip_ref.extractall("temp/")
    
    ic = IceCube(matfile)    
    
    ilat = ic.lat
    ilsm = np.squeeze(ic.get_data("stype"))
    iiwp = np.squeeze(ic.get_data("iwp"))
    dlat = xml.load("temp/lat_grid.xml")
    snow = np.squeeze(xml.load("temp/snow_depth.xml"))
    ice  = np.squeeze(xml.load("temp/sea_ice_cover.xml"))
    
    
    lsm = np.squeeze(xml.load("temp/lsm.xml"))
    iwc = np.squeeze(xml.load("temp/iwc.xml"))
    z1   = np.squeeze(xml.load("temp/z_field.xml"))
    p   = np.squeeze(xml.load("temp/p_grid.xml"))
    z   = pres2alt(p)
    iwp = np.zeros(lsm.shape)
    iwp1 = np.zeros(lsm.shape)
    for i in range(iwc.shape[1]):
        iwp[i] = np.trapz(iwc[:, i], z)
    
        
    fig, ax = plt.subplots(1,1, figsize = [8,8])
    
    ax.plot(ilat, iiwp, label = "IceCube")
    ax.plot(dlat, iwp, label = "DARDAR")
    ax.legend()
    
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.plot(ilat, ic.wvp )
    ax.set_xlabel("lat [deg]")
    ax.set_ylabel("wvp []")
    plt.show()


    args = []
    for i in range(ilat.shape[0]):
        diff = np.abs(dlat - ilat[i])
        iarg = np.argmin(diff)
        args.append(iarg)
        
    dlat_sub = dlat[args]    
    iwp_sub  = iwp[args]
    
    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    ax.scatter(iwp_sub, iiwp)
    ax.set_xlabel("IWP DARDAR [kg/m2]")
    ax.set_ylabel("IWP IceCube [kg/m2]")


#%% draw the PDF of the IceCube Ta
def get_pdf(matfiles):
    """
    Compute the PDF of IceCube Ta
    every 5th simulation is chosen to represent the IceCube resolution @ 15 km

    Parameters
    ----------
    matfiles : input *.mat files

    Returns
    -------
    None.

    """
    Ta = []
    for file in matfiles[:]:
     #   print (file)
        
        ic = IceCube(file)
        
        try:
            wvp = ic.wvp         
        except:
            print ("wvp not availble ", file)
        
        if np.sum(wvp) == 0:
                shutil.move(file, '/home/inderpreet/data/temp/OutIcube/')       
        
     #   print (ic.lat.shape, ic.tb.shape)
        
        lat2, i2, i1 = np.unique(ic.lat, return_index=True, return_inverse=True)
    
        tb = ic.tb
        tb = tb.reshape(-1, 1)
        yf = gaussfilter(lat2, tb[i2], 15/111)
        
        ta = yf[i1]
        Ta.append(ta)
        
        
    Ta = np.concatenate(Ta, axis = 0)    
    Ta = np.squeeze(Ta)

    # take every fifth simulation, IceCube has 15km resolution
    Ta = Ta[::5]
    
    bins = np.arange(90, 300, 1.5)
    
    ic_all = IceCube(matfiles)
    tb = ic_all.tb
    hist_a = np.histogram(Ta,  bins, density = True)
    
        
    fig, ax = plt.subplots(1,1, figsize = [8, 8])
    ax.plot(bins[:-1], hist_a[0])
    
    ax.set_yscale('log')
    ax.set_ylabel('PDF [#/K]')
    ax.set_xlabel('Ta [K]')
    plt.show()
    fig.savefig("/home/inderpreet/git/Projects/IWC2TB/WorkArea/IceCube/Figures/PDF_icecube.pdf", bbox_inches = "tight")
    
 #   plt.plot(ic.tb)
 
#%%  
# gauss filtering with python *gaussfilter*    
def compare_ta_tb(matfile):
    """
    compares ta and tb

    Parameters
    ----------
    matfile : input *.mat file

    Returns
    -------
    None.

    """
    ic = IceCube(matfile)
    
    lat2, i2, i1 = np.unique(ic.lat, return_index=True, return_inverse=True)
    
    tb = ic.tb
    tb = tb.reshape(-1, 1)
    yf = gaussfilter(lat2, tb[i2], 15/111)
    
    t_p = yf[i1]
    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    ax.plot(ic.lat,ic.tb, label = "tb")
    ax.plot(ic.lat,t_p, label = "ta")
    ax.set_xlabel("latitude [deg]")
    ax.set_ylabel("brightness temp [K]")
    ax.legend()   
    plt.show() 

#%%
# calculate IWP 
def get_total_iwp(matfiles):  
    """
    compute IWP from DARDAR_ERA *.zip file

    Parameters
    ----------
    matfiles : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ic = IceCube(matfiles)
    
    lat = ic.lat
    iwp = ic.iwp
    
    latbins  = np.arange(-30, 30.5, 0.5)
    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)/nbins
    
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.plot(iwp_mean, latbins)
    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("Lat [deg]") 
    fig.savefig("IWP_IceCube.png", bbox_inches = "tight")
    

#%%
def apply_gaussfilter(x, y, w):   
    """
    apply gauss filter to compute the antenna weighted values for 
    any paramter simulated by ARTS

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    yfilter : TYPE
        DESCRIPTION.

    """
    
    x2, i2, i1 = np.unique(x, return_index=True, return_inverse=True)
    
    y = y.reshape(-1, 1)
    yf = gaussfilter(x2, y[i2], w)
    
    yfilter = yf[i1]
    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    ax.plot(x, y, label = "tb")
    ax.plot(x, yfilter, label = "ta")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()   
    plt.show()     
    return yfilter
    

#%%
if __name__ == '__main__':    
        
    #%%
        
    inpath =  os.path.expanduser('~/Dendrite/Projects/IWP/IceCube/IceCube_m30_p30_2mom')  
     
    #matfile = glob.glob("/home/inderpreet/data/temp/OutIcube/*.mat")
    matfiles = glob.glob(os.path.join(inpath, "*.mat"))
    
    dardarpath = os.path.expanduser("~/Dendrite/Projects/IWP/IceCube/DARDAR_ERA_m30_p30_N0star")
    
    
    #%%   
    
    get_pdf(matfiles)
