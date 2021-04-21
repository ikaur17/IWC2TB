#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:02:32 2021

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
from matplotlib import ticker, cm
from iwc2tb.GMI.remove_oversampling_gmi import remove_oversampling_gmi
from iwc2tb.GMI.swap_gmi_183 import swap_gmi_183
from iwc2tb.GMI.three_sigma_rule import three_sigma_rule
from iwc2tb.common.hist2d import hist2d
from iwc2tb.GMI.filter_gmi_sat import filter_gmi_sat
plt.rcParams.update({'font.size': 30})
   
#%%plot PDFs for all 4 channels   
def plot_pdf_gmi(Ta, Tb, bins= None, figname = "distribution_gmi.pdf"):
    
    if bins is None:
        bins = np.arange(100, 310, 2)
        
    fig, axs = plt.subplots(2,1, figsize = [20, 20])
    fig.tight_layout(pad=3.0)
    
    for i, ax in enumerate(fig.axes):
        
        if i == 1:
            i = 2
    
        hist_a = np.histogram(Ta[::3, i],  bins, density = True)      
        hist_b = np.histogram(Tb[::10, i],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0],'b', label =  freq[i]+ ' obs', linewidth = 2, alpha = 0.5)       
        ax.plot(bins[:-1], hist_a[0],'b--', label =  freq[i] + ' sim', linewidth = 2, alpha = 0.5)
        
        hist_a = np.histogram(Ta[::3, i+1],  bins, density = True)      
        hist_b = np.histogram(Tb[::10, i+1],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0], 'r', label =  freq[i+1] + ' obs', linewidth = 2, alpha = 0.5)       
        ax.plot(bins[:-1], hist_a[0], 'r--', label =  freq[i+1] +  ' sim', linewidth = 2, alpha = 0.5)

#        ax.set_title(freq[i] + " GHz")
    
        ax.set_yscale('log')
        ax.set_ylabel('PDF [#/K]')
        ax.set_xlabel('Ta [K]')
        ax.legend()
    fig.savefig("Figures/" + figname, bbox_inches = "tight")


#%% scatter plots

def plot_scatter(Ta, Tb_gmi,  freq, figname = "scatter_gmi.pdf"):
    fig, axs = plt.subplots(2, 3, figsize = [30, 20])
    fig.tight_layout(pad=3.0)
    for ix, i in enumerate(range(0, 3)):

        for jx, j in enumerate(range(i+1, 4)):

            if ix == 2:
                jx = 2
                ix = 1
   
#            axs[ix, jx].scatter(Ta[:, i], Ta[:, j], label = 'simulated', alpha = 0.1)
            axs[ix, jx].scatter(Tb_gmi[:, i], Tb_gmi[:, j], label = 'observed', alpha = 0.3) 
            axs[ix, jx].scatter(Ta[:, i], Ta[:, j], label = 'simulated', alpha = 0.3)

            axs[ix, jx].set_xlabel(freq[i] + " [GHz]")
            axs[ix, jx].set_ylabel(freq[j] + " [GHz]")
            axs[ix, jx].legend()
    
    fig.savefig("Figures/" + figname, bbox_inches = "tight" )       
            
    
    
#%%
def plot_hist2d(ta, tb0, figname = "hist2d.png"):
    
    pd = ta[:, 0] - ta[:, 1]
    
    pd_sampled = pd[::3]
    ta_sampled = ta[::3, 0]
    
    pd_sampled_gmi = (tb0[:, 0] - tb0[:, 1])
    ta_sampled_gmi = tb0[:, 0]
    
    fig, ax = plt.subplots(1, 1, figsize = [15, 15])
    xbins = np.arange(100, 310, 2)
    ybins = np.arange(-5, 60, 1)
    # ybins1  = np.arange(-5, 5, 0.5)
    # ybins2 = np.arange(5, 20, 0.75)
    # ybins3 = np.arange(20, 30, 1.25)
    # ybins4 = np.arange(30, 60, 2)
    # ybins = np.concatenate([ybins1, ybins2, ybins3, ybins4])
    counts, xbins, ybins=np.histogram2d(ta_sampled, pd_sampled,  
                                        bins=(xbins, ybins))
# make the contour plot

    cs = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'blue', locator=ticker.LogLocator(), alpha = 0.5))

    
    
        
    counts, xbins, ybins=np.histogram2d(ta_sampled_gmi, pd_sampled_gmi,
                                        bins=(xbins, ybins))
# make the contour plot

    cs_gmi  = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red',locator=ticker.LogLocator(),  alpha = 0.5))
    
    lines = [ cs.collections[0], cs_gmi.collections[0]]
#    labels = ['CS1_neg','CS1_pos','CS2_neg','CS2_pos']
    plt.legend(lines, ["simulated", "observed"], loc = 'upper left')
    ax.set_xlabel(" Brightness temperature 166 V [K] ")
    ax.set_ylabel("Polarisation difference [V-H] [K]")


    # fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    # ax.scatter(ta_sampled_gmi, pd_sampled_gmi, label = "observed", alpha = 0.3)
    # ax.scatter( ta_sampled, pd_sampled, label = "simulated", alpha = 0.3)

    # ax.set_xlabel(" Brightness temperature 166 V [K] ")
    # ax.set_ylabel("Polarisation difference [V-H] [K]")
    # ax.legend()

    fig.savefig("Figures/" + figname, bbox_inches = "tight")
#%%

#%%



#%%
def call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims = None, stype_sim = None, stype_gmi = None, 
                figname = None):

        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
#    ta, lat, lon, stype  = get_Ta(matfiles, latlims, lsm = None)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    
    plot_pdf_gmi(ta1, tb0, bins= None, figname = "distribution_gmi.pdf")
    
#    ta1 = swap_gmi_183(ta1)

#    plot_hist2d(ta1, tb0, figname) 

    plot_hist(ta1, tb0, figname)
#%%
def plot_hist(ta, tb, figname = "contour2d.png"):
    from matplotlib import ticker
    
    fig, ax = plt.subplots(1, 1, figsize = [12, 12])
    
    xdat = ta[:, 0]
    ydat = ta[:, 0] -  ta[:, 1]
    
#    xyrange = [[xdat.min()-5, xdat.max()+5],[ydat.min()-5, ydat.max()+ 5]] # data range
    xyrange = [[100, 300], [-5, 60]] # data range
  
    bins = [100, 65] # number of bins
    thresh = 1/xdat.shape[0] * 2  #density threshold
    
    
    # histogram the data
    hh, locx, locy = np.histogram2d(ta[:, 0], ta[:, 0] - ta[:, 1], 
                                    range=xyrange, bins=bins, density = True)
    posx = np.digitize(ta[:, 0], locx)
    posy = np.digitize(ta[:, 0] - ta[:, 1], locy)
    xdat = ta[:, 0]
    ydat = ta[:, 0] - ta[:, 1]
    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    #hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
    
    cs = ax.contour(np.flipud(hh.T),colors= 'red',
                    extent=np.array(xyrange).flatten(), 
                locator= ticker.LogLocator(), origin='upper')
#    plt.colorbar()   
    ax.plot(xdat1, ydat1, '.',color='red', alpha = 0.2)
    
    hh, locx, locy = np.histogram2d(tb[:, 0], tb[:, 0] - tb[:, 1], 
                                    range=xyrange, bins=bins, density = True)
    posx = np.digitize(tb[:, 0], locx)
    posy = np.digitize(tb[:, 0] - tb[:, 1], locy)
    xdat = tb[:, 0]
    ydat = tb[:, 0] - tb[:, 1]
    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    #hh[hh < thresh] = np.nan # fill the areas with low density by NaNs
    
    cs_gmi = ax.contour(np.flipud(hh.T),colors = 'blue',
                        extent=np.array(xyrange).flatten(), 
                locator=ticker.LogLocator(),  origin='upper')
 #  plt.colorbar()   
    ax.plot(xdat1, ydat1, '.',color='blue',  alpha = 0.2)
    lines = [ cs.collections[0], cs_gmi.collections[0]]
#    labels = ['CS1_neg','CS1_pos','CS2_neg','CS2_pos']
    plt.legend(lines, ["simulated", "observed"], loc = 'upper left')
    ax.set_xlabel(" Brightness temperature 166 V [K] ")
    ax.set_ylabel("Polarisation difference [V-H] [K]")

    fig.savefig("Figures/" + figname, bbox_inches = "tight")    
   
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
def apply_three_sigma(ta):
    
    
    tx = three_sigma_rule(ta[:, 3])
    
    itx = ta[:, 2] < tx
    
    
    return itx
#%%

def filter_lat_lon(lat, lon, latlims, lonlims):

    
    lamask = (lat > latlims[0]) & (lat < latlims[1])
    lomask = (lon > lonlims[0]) & (lon < lonlims[1])
    
    lmask = np.logical_and(lamask, lomask)
    
    return lmask

#%%
def filter_lsm(stype, lsm):
    mask2 = np.ones(stype.shape, dtype = bool)  
    if np.isscalar(lsm):        
        mask2 = stype == lsm             
    else:
        for i in range(len(lsm)):
            im1 = stype == lsm[i]
            mask2  = np.logical_or(im1, mask2)    
            
    return mask2        
    
#%%    
    

if __name__ == "__main__":
    
    # GMI satellite data   
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    #inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1C/2019/01/')
    gmifiles = glob.glob(os.path.join(inpath, "*/*.HDF5"))
    
    random.shuffle(gmifiles)
    gmi_sat = GMI_Sat(gmifiles[:25])
    
    lat_gmi = gmi_sat.lat
    lon_gmi = gmi_sat.lon
    tb_gmi  = gmi_sat.tb
    lsm_gmi = gmi_sat.get_lsm()
    iwp_gmi = gmi_sat.get_gprofdata('iceWaterPath')
    
    
    lat_gmi = lat_gmi.ravel()
    lon_gmi = lon_gmi.ravel()
    tb_gmi  = tb_gmi.reshape(-1, 4)
    lsm_gmi = lsm_gmi.ravel()
    iwp_gmi = iwp_gmi.ravel()
        
    tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi = remove_oversampling_gmi(tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi)
 
#%%    
    # GMI simulations    
    inpath    =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_f07')  
    inpath1   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_si') 
 
     
    matfiles = glob.glob(os.path.join(inpath, "2010_0*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath1, "2010_*.mat"))

    
    matfiles = matfiles + matfiles1 
    
    gmi      = GMI(matfiles)
    ta       = gmi.ta_noise
    lat, lon = gmi.lat, gmi.lon
    iwp      = gmi.iwp
    stype    = gmi.stype
    
    ta = swap_gmi_183(ta)

       
    # GMI frequencies
    freq     = ["166.5V", "166.5H", "183+-3", "183+-7"]

#%%  , 


    latrange = np.arange(-10, 21, 10)
    
    latrange = [55, 65]
    xyrange = [ [100, 280], [-10, 50]]
    
    for i in range(0, 1):
        
        latlims = [latrange[i], latrange[i+1]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [8, 8])
        
        ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 0 ) 
        
        itx = apply_three_sigma(ta1)
        
        #itx = np.ones(ta1.shape[0], "bool")
        
        #itx = ~itx

        pd  = ta1[itx, 0] - ta1[itx, 1]
        hh, xyrange, xdat1, ydat1 = hist2d(ta1[itx, 0], pd, xyrange, [50, 50])
        cs = ax1.contourf(np.flipud(hh.T), cmap= 'Blues',
                extent=np.array(xyrange).flatten(), 
            locator= ticker.LogLocator(), origin='upper')
        cbar = fig.colorbar(cs, ax = ax1)  
        ax1.plot(xdat1, ydat1, '.',color='blue', alpha = 0.2)
        ax1.set(ylim = [-5, 50], xlim = [100, 300])
        
        tb1, glat1, glon1 = filter_gmi_sat(lat_gmi, lon_gmi, tb_gmi, lsm_gmi,  latlims, lsm = 1) 
        itx_g = apply_three_sigma(tb1) 
        
        #itx_g = np.ones(tb1.shape[0], "bool")
        
        #itx_g = ~itx_g
        pd  = tb1[itx_g, 0] - tb1[itx_g, 1]
        hh, xyrange, xdat1, ydat1 = hist2d(tb1[itx_g, 0], pd, xyrange, [50, 50])
        cs = ax2.contourf(np.flipud(hh.T), cmap= 'Blues',
                extent=np.array(xyrange).flatten(), 
            locator= ticker.LogLocator(), origin='upper')
        cbar = fig.colorbar(cs, ax = ax2)  
        ax2.plot(xdat1, ydat1, '.',color='blue', alpha = 0.2)
        ax2.set(ylim = [-5, 50], xlim = [100, 300])
        
        ax1.set_title(str(latlims[0]) + "-" + str(latlims[1]))
        


#%% # amazon
        
    xyrange = [ [100, 280], [-10, 40]]
        
    # latlims = [-15, 0]        
    # lonlims = [288, 307 ]
    
    # africa
    latlims = [-4, 19]        
    lonlims = [15 , 31]
    
    
    #N America
    #latlims = [33, 52]        
    #lonlims = [244, 278]
    
    lon_gmi = lon_gmi%360
    
    lmask = filter_lat_lon(lat, lon, latlims, lonlims)
    gmask = filter_lat_lon(lat_gmi, lon_gmi, latlims, lonlims)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [8, 16])
    pd  = ta[lmask, 0] - ta[lmask, 1]
    hh, xyrange, xdat1, ydat1 = hist2d(ta[lmask, 0], pd, xyrange, [25, 50])
    cs = ax1.contourf(np.flipud(hh.T), cmap= 'Blues',
            extent=np.array(xyrange).flatten(), 
        locator= ticker.LogLocator(), origin='upper')
    cbar = fig.colorbar(cs, ax = ax1)  
    ax1.plot(xdat1, ydat1, '.',color='blue', alpha = 0.2)
    ax1.set(ylim = [-5, 30], xlim = [100, 300])
    ax1.set_title('simulations')

    
    #itx_g = ~itx_g
    #smask = filter_lsm(lsm_gmi, [3, 4, 5, 6, 7])
    #gmask = np.logical_and(smask, gmask)
    pd  = tb_gmi[gmask, 0] - tb_gmi[gmask, 1]
    
    hh, xyrange, xdat1, ydat1 = hist2d(tb_gmi[gmask, 0][::5], pd[::5], xyrange, [50, 50])
    cs = ax2.contourf(np.flipud(hh.T), cmap= 'Blues',
            extent=np.array(xyrange).flatten(), 
        locator= ticker.LogLocator(), origin='upper')
    cbar = fig.colorbar(cs, ax = ax2)  
    ax2.plot(xdat1, ydat1, '.',color='blue', alpha = 0.2)
    ax2.set(ylim = [-5, 30], xlim = [100, 300])
    ax2.set_title('observations')
    
    fig.suptitle('Amazon') 
    fig.savefig("Amazon.png", bbox_inches = "tight")
    ip = pd < 0
    
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    pd  = tb_gmi[gmask, 0] - tb_gmi[gmask, 1]
    ax.scatter(tb_gmi[gmask, 0], pd)
    ax.scatter(ta[lmask, 0][::6], ta[lmask, 0][::6] - ta[lmask, 1][::6], label = "Simulated")
    
    #
    #ax.scatter(tb_gmi[gmask, 0], pd, label = "GMI")   
    ax.legend()
#%%    
    
    
    
    

    
    
    
    
    
    

    
        