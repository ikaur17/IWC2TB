#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:02:32 2021

This script makes plots for GMI simulations from ARTS

This script has functions for :
    1. calculating Ta
    2. plotting scatter b/w different channels
    3. plot PDF for all four channels
    
    4. has provision to subset data acording to latitude and stype

@author: inderpreet
"""

import os
import numpy as np
import glob
from iwc2tb.py_atmlab.apply_gaussfilter import apply_gaussfilter
from iwc2tb.GMI.GMI import GMI
from iwc2tb.GMI.GMI_SatData import GMI_Sat
import random
import matplotlib.pyplot as plt
from iwc2tb.common.add_gaussian_noise import add_gaussian_noise
from iwc2tb.py_atmlab.gaussfilter import filter_stype
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from iwc2tb.common.plot_locations_map import plot_locations_map
import scipy
import pickle
from matplotlib import ticker, cm
from plot_pdf_gmi import plot_pdf_gmi
from plot_scatter import plot_scatter
from plot_hist2d import plot_hist2d
from filter_gmi_sat import filter_gmi_sat
from remove_oversampling_gmi import remove_oversampling_gmi
from plot_hist import plot_hist
plt.rcParams.update({'font.size': 20})

#%%
def call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims = None, stype_sim = None, stype_gmi = None, 
                figname = None):

        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    
    plot_pdf_gmi(ta1, tb0, bins= None, figname = "distribution_gmi.pdf")
    

    plot_hist(ta1, tb0, figname)
    
#%%
def call_hist2d_all(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims = None, stype_sim = None, stype_gmi = None, 
                figname = None):

        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    

    fig, ax = plt.subplots(1, 2, figsize = [16, 8])
    ax = ax.ravel()
    

    #water  ------------------------------- 
    stype_gmi = [1, 12]
    stype_sim = 0
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    

    plot_hist(ta1, tb0, figname, ax = ax[0]) 
    
    #land  ----------------------------------  
    stype_gmi = [3, 4, 5, 6, 7]
    stype_sim = [1, 4]
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    

    plot_hist(ta1, tb0, figname, ax = ax[1]) 

    # #snow ------------------------------------  
    # stype_gmi = [8, 9, 10, 11]
    # stype_sim = [2, 5, 7, 9]
    # tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
    #                                   tb_gmi[::6, :], lsm_gmi[::6],
    #                                   latlims, stype_gmi))
    
    # ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)

    # plot_hist(ta1, tb0, figname, ax = ax[2]) 

    # #seaice   ---------------------------------
    # stype_gmi = [2,14]
    # stype_sim = [3,6]
    # tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
    #                                   tb_gmi[::6, :], lsm_gmi[::6],
    #                                   latlims, stype_gmi))
    # ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    

    # plot_hist(ta1, tb0, figname, ax = ax[3]) 
    
    
    ax[0].set_xlabel(" Brightness temperature 166V GHz [K] ")
    ax[0].set_ylabel("Polarisation difference [K]")
    
    ax[1].set_xlabel(" Brightness temperature 166V GHz [K] ")
    #ax[0].set_ylabel("Polarisation difference [K]")
    
    
    titles = ["Water", "Land"]
    for i in range(2):
        ax[i].grid("on", alpha = 0.3)
        ax[i].set_title(titles[i])
        ax[i].set_ylim([-7, 55])
        ax[i].set_xlim([100, 310])
    ax[0].text(102, 56, "a)")
    ax[1].text(102, 56, "b)")
        
    fig.savefig("hist2d_land_water_jan.pdf", bbox_inches = "tight")    
#%%
def compare_psd(ta, lat, lon, stype, ta1, lat1, lon1, stype1, latlims):
    
    bins = np.arange(100, 300, 1)
    t, la, lo = filter_gmi_sat(lat, lon, ta, stype, latlims)      
    t1, la1, lo1 = filter_gmi_sat(lat1, lon1, ta1, stype1, latlims)
    
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.hist(t[:, 2], bins, density = True, label = "f07", histtype = "step")
    ax.hist(t1[:, 2], bins, density = True, label = "DARDAR", histtype = "step")
    ax.legend()
    ax.set_yscale('log')
    
#%%
def call_hist2d_snow(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims = None, stype_sim = None, stype_gmi = None, 
                figname = None):

        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    

    fig, ax = plt.subplots(1, 2, figsize = [16, 8])
    ax = ax.ravel()
    

    #snow ------------------------------------  
    stype_gmi = [8, 9, 10, 11]
    stype_sim = [2, 5, 7, 9]
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)

    plot_hist(ta1, tb0, figname, ax = ax[0]) 
    


    #seaice   ---------------------------------
    stype_gmi = [2, 14]
    stype_sim = [3, 6]
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    
    
    plot_hist(ta1, tb0, figname, ax = ax[1]) 
    
    
    ax[0].set_xlabel(" Brightness temperature 166V GHz [K] ")
    ax[0].set_ylabel("Polarisation difference [K]")
    
    ax[1].set_xlabel(" Brightness temperature 166V GHz [K] ")
    #ax[0].set_ylabel("Polarisation difference [K]")
    
    
    titles = [ "Snow", "Seaice"]
    for i in range(2):
        ax[i].grid("on", alpha = 0.3)
        ax[i].set_title(titles[i])
        ax[i].set_ylim([-7, 55])
        ax[i].set_xlim([100, 310])
        

    im = tb0[:, 0] - tb0[:, 1] > 20
    plot_locations_map(lat0[im], lon0[im])  
    
    ax[0].text(102, 56, "a)")
    ax[1].text(102, 56, "b)")     
        
    fig.savefig("hist2d_snow_surface_jan.pdf", bbox_inches = "tight")    
    

#%%
def compare_psd(ta, lat, lon, stype, ta1, lat1, lon1, stype1, latlims):
    
    bins = np.arange(100, 300, 1)
    t, la, lo = filter_gmi_sat(lat, lon, ta, stype, latlims)      
    t1, la1, lo1 = filter_gmi_sat(lat1, lon1, ta1, stype1, latlims)
    
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.hist(t[:, 2], bins, density = True, label = "f07", histtype = "step")
    ax.hist(t1[:, 2], bins, density = True, label = "DARDAR", histtype = "step")
    ax.legend()
    ax.set_yscale('log')
        
#%%

def swap_gmi_183(ta1):
    
        temp = np.zeros(ta1.shape)
        temp[:,  0] = ta1[:, 0]
        temp[:,  1] = ta1[:, 1]
        temp[:,  2] = ta1[:, 3]
        temp[:,  3] = ta1[:, 2]
        ta1 = temp.copy()
        
        return ta1 
    

#%%    
    

if __name__ == "__main__":
    
#%% set input paths 

    # GMI satellite data   
    inpath_gmi   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2020/01/')
    # GMI simulations    
    inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/') 
    key = "lpa"
    # GMI frquencies
    freq     = ["166.5V", "166.5H",  "183+-7", "183+-3",]    

#%% read GMI measurrments 
    gmifiles = glob.glob(os.path.join(inpath_gmi, "*/*.HDF5"))
    
    random.shuffle(gmifiles)
    gmi_sat = GMI_Sat(gmifiles[:25])
    
    lat_gmi = gmi_sat.lat
    lon_gmi = gmi_sat.lon
    tb_gmi  = gmi_sat.tb
    lsm_gmi = gmi_sat.get_lsm()
    iwp_gmi = gmi_sat.iwp
    
    
    lat_gmi = lat_gmi.ravel()
    lon_gmi = lon_gmi.ravel()
    tb_gmi  = tb_gmi.reshape(-1, 4)
    lsm_gmi = lsm_gmi.ravel()
    iwp_gmi = iwp_gmi.ravel()
    
    
    tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi = remove_oversampling_gmi(tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi)
    
    
    tb_gmi = tb_gmi[::5, :]
    lat_gmi= lat_gmi[::5]
    lon_gmi = lon_gmi[::5] 
    lsm_gmi = lsm_gmi[::5]
    tbbins = np.arange(100, 310, 2)
    hist_gmi = np.zeros([tbbins.shape[0] - 1, 4])
     
    for i in range(4):
        hist_gmi[:, i], _ = np.histogram(tb_gmi[:, i], tbbins, density = True)
        
    

    
    hist_gmi1 = np.zeros([tbbins.shape[0] - 1, 4])  
    mask  = np.abs(lat_gmi) <= 30.0  
    for i in range(4):
        hist_gmi1[:, i], _ = np.histogram(tb_gmi[mask, i], tbbins, density = True)
        
    hist_gmi2 = np.zeros([tbbins.shape[0] - 1, 4])
    mask  = (np.abs(lat_gmi) > 30.0)  & (np.abs(lat_gmi) <= 45)
    for i in range(4):
        hist_gmi2[:, i], _ = np.histogram(tb_gmi[mask, i], tbbins, density = True)       
        
    hist_gmi3 = np.zeros([tbbins.shape[0] - 1, 4])
    mask  = np.abs(lat_gmi) > 45.0 
    for i in range(4):
        hist_gmi3[:, i], _ = np.histogram(tb_gmi[mask, i], tbbins, density = True)          
        
    
    
    with open("hist_gmi_jan.pickle", "wb") as f:
        pickle.dump(hist_gmi, f)
        pickle.dump(hist_gmi1, f)
        pickle.dump(hist_gmi2, f)
        pickle.dump(hist_gmi3, f)
        pickle.dump(tbbins, f)
        f.close()

#%%

#%% read GMI simulations 
     
    matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
    matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
    matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
    #matfiles = matfiles1
    matfiles = matfiles1 + matfiles2 + matfiles3
    gmi                 = GMI(matfiles)
    ta, lat, lon, stype = gmi.ta_noise, gmi.lat, gmi.lon, gmi.stype
    ta                  = swap_gmi_183(ta)
    
#%% PDF of simulations
    
    hist = np.zeros([tbbins.shape[0] - 1, 4])
     
    for i in range(4):
        hist[:, i], _ = np.histogram(ta[:, i], tbbins, density = True)
        
        
    hist1 = np.zeros([tbbins.shape[0] - 1, 4])  
    mask  = np.abs(lat) <= 30.0  
    for i in range(4):
        hist1[:, i], _ = np.histogram(ta[mask, i], tbbins, density = True)
        
    hist2 = np.zeros([tbbins.shape[0] - 1, 4])
    mask  = (np.abs(lat) > 30.0)  & (np.abs(lat) <= 45)
    for i in range(4):
        hist2[:, i], _ = np.histogram(ta[mask, i], tbbins, density = True)       
        
    hist3 = np.zeros([tbbins.shape[0] - 1, 4])
    mask  = np.abs(lat) > 45.0 
    for i in range(4):
        hist3[:, i], _ = np.histogram(ta[mask, i], tbbins, density = True)          
        
    

    with open("hist_lpa_jan.pickle", "wb") as f:
        pickle.dump(hist, f)
        pickle.dump(hist1, f)
        pickle.dump(hist2, f)
        pickle.dump(hist3, f)
        pickle.dump(tbbins, f)
        f.close()

#%%
    cmask = gmi.iwp >= 1
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])

    ax.scatter(ta[cmask, 0], ta[cmask, 0] - ta[cmask, 1])
    
#%% all surface types in one plot
    

    latlims  = [0, 65]
    lsm = None   
    call_hist2d_all(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_all_" + key + ".pdf")
    

    latlims  = [45, 65]
    lsm = None   
    call_hist2d_snow(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_snow_" + key + ".pdf")

   
#%% all surface types
    #Tropics    
    print ("doing tropics, all")
    latlims  = [0, 30]
    lsm = None    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_tropics_all_" + key + ".pdf")
 
    # higher latitudes
    
    print ("doing 30-45, all")    
    latlims  = [30, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm,  
                figname = "hist2d_gmi_30-45_all_" + key + ".pdf")

    
    print ("doing 45-60, all")
    
    latlims  = [45, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_45-60_all_" + key + ".pdf")


#%% tropics land    
    print ("doing tropics, land")

    lon1 = 73
    lon2 = 107
    
    lat1 = 30
    lat2 = 55
    
    lon = lon%360
    
    m1 = (lon > lon1) & (lon < lon2)
    m2 = (lat > lat1) & (lat < lat2)
    
    mask = np.logical_and(m1, m2)
    
    lon_gmi = lon_gmi%360
    
    m1 = (lon_gmi > lon1) & (lon_gmi < lon2)
    m2 = (lat_gmi > lat1) & (lat_gmi < lat2)
    
    mask_gmi = np.logical_and(m1, m2)


    
    stype_gmi = [3, 4, 5, 6, 7]
    stype_sim = [1, 4]
        
    latlims  = [10, 30]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_tropics_land_" + key + ".pdf")        

    # higher latitudes land
    
    print ("doing high lats, land")
    
    latlims  = [30, 45]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_land_" + key + ".pdf")
    
    print ("doing high lats, land")


    
    latlims  = [45, 65]   
        
  
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-65_land_" + key + ".pdf")

#%%    
    latlims  = [0, 65]  
    stype_gmi = [1, 3, 4, 5, 6, 7]
    stype_sim = [1, 4]
        
  
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_all_land_" + key + ".pdf")
    

    
#%% higher latitudes sea
    
    print ("doing high lats, sea")
    
    stype_gmi = [1, 12]
    stype_sim = 0
    
    latlims  = [-30, 0]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_sea_" + key + ".pdf")
    
 
    print ("doing high lats, sea")
    
    latlims  = [45, 65]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-65_sea_" + key + ".pdf")        
    
    # tropics sea
    
    print ("doing tropics, sea")

    latlims  = [0, 30]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi,  
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_tropics_sea_" + key + ".pdf")  
    
    print ("All, sea")

    latlims  = [0, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi,  
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_all_sea_" + key + ".pdf")  


    
    
#%% higher latitudes sea-ice
    
    print ("highlats, sea-ice")
    
    stype_gmi = [2,14]
    stype_sim = [3,6]
    
    latlims  = [50, 65]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_highlat_sea-ice_" + key + ".pdf") 
    
#%%
    print ("highlats, snow")
    stype_gmi = [8, 9, 10, 11]
    stype_sim = [2, 5, 7, 9]
    
    latlims  = [30, 45]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_snow_" + key + ".pdf")           
    
    print ("highlats, snow")
    latlims  = [45, 65]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-60_snow_" + key + ".pdf")  

#%%
    latlims  = [0, 65]  
    stype_gmi = [2, 14,8, 9, 10, 11]
    stype_sim = [3, 6,2, 5, 7, 9]
        
  
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_snowseaice_land_" + key + ".pdf")    

#%%    

    plot_pdf_gmi(ta, tb_gmi, bins= None, figname = "distribution_gmi.pdf")

#%%     
    latlims = [0, 65]
    

    stype_sim = None
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)   
    
    fig, ax = plt.subplots(1, 1, figsize = [15, 10])
    
    bins = np.arange(-5, 50, 0.2)
    hist = np.histogram(ta1[:, 0] - ta1[:, 1], bins, density = True) 
    ax.step(hist[1][:-1], hist[0], 'r', label = "simulated")    
    hist_gmi = np.histogram(tb_gmi[::6, 0] - tb_gmi[::6, 1], bins, density = True) 
    ax.step(hist_gmi[1][:-1], hist_gmi[0], 'b', label = "observed")    
    ax.legend()   
    ax.set_yscale('log')
    ax.set_xlabel("Polarisation difference 166 GHz [V-H]")
    ax.set_ylabel("PDF [#/K]")
    fig.savefig("Figures/PD_PDF.pdf", bbox_inches = "tight")
    


#%%
    with open("PD_" + key + ".pickle", 'wb') as f:
        pickle.dump(hist_gmi[0],f)
        pickle.dump(hist[0], f)
        pickle.dump(bins, f)
        f.close()
        
#%%  
    latlims = [30, 45]
    stype_sim = [3, 4, 5, 6, 7]
    ta1, lat1, lon1 = filter_gmi_sat(lat_gmi, lon_gmi, tb_gmi, lsm_gmi, latlims, stype_sim)      
    m1 = ta1[:, 0] > 200
    m2 = ta1[:, 0] - ta1[:, 1] > 40
    m = np.logical_and(m1, m2)
    
    plot_locations_map(lat1[m], lon1[m])
    

    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    bins = np.arange(-5, 40, 1)
    ax.hist(ta1[:, 0] - ta1[:, 1], bins, density = True, histtype = 'step')
    
    
    
#%%
    latlims = [45, 65]
    stype_sim = [2, 5, 7, 9]
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)    
    m1 = ta1[:, 0] > 280
    m2 = ta1[:, 0] - ta1[:, 1] > 0
    
    m = np.logical_and(m1, m2)

    ax.hist(ta1[:, 0] - ta1[:, 1], bins, density = True,histtype = 'step')
    ax.set_yscale('log')
    plot_locations_map(lat1[m], lon1[m])
    

  



    
    
    
    
    
    
    

    
        