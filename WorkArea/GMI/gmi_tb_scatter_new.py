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
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 50})




#%%
#get antenna weighted TB values
def get_Ta(matfiles,  latlims = None, lsm = None ):
    """
    Get antenna weighted brightness temperature values
    Also, allows for sub-setting the data by lat limits and stype.
    

    Parameters
    ----------
    matfiles : list containing names of files for GMI simulations
    latlims : list, optional
        THe lower and upper limit of lat lims. 
        Only positive values need to be given. 
        The default is None.
    lsm : float, optional
        the stype for which Ta has to be subset for. The default is None.

    Returns
    -------
    Ta_all : np.array [n x 4], antenna weighted TB for 4 GMI channels
    lat_all : np.array [n], the latitudes where Ta_all are defined

    """
     
    Ta_all  = []

#    for i in range(4):

    lat_all = []
    lon_all = []
    stype_all = []
    for ix, file in enumerate(matfiles[:]):
            print (file)
            
            gmi = GMI(file)
            
            lat_all.append(gmi.lat)
            lon_all.append(gmi.lon)
            
            lsm = apply_gaussfilter(gmi.lat, gmi.stype, 6/111)

            
            # re-classfiy sea-ice classified as snow
            ix = np.where(np.logical_and(lsm[:, 0] == 2, gmi.stype == 3))
            lsm[ix, 0] == 0
            
            # re-classfiy sea-ice classified as land
            ix = np.where(np.logical_and(lsm[:, 0] == 1, gmi.stype == 3))
            lsm[ix, 0] == 0
            
            # re-classify snow classified as water
            ix = np.where(np.logical_and(lsm[:, 0] == 0, gmi.stype == 2))
            lsm[ix, 0] == 1
                
            lsm = np.squeeze(lsm)
            print (lsm.shape)
            stype_all.append(lsm)
            
            Ta = []
           
            for i in range(4):
            # calculate antenna weighted values
                ta = apply_gaussfilter(gmi.lat, gmi.tb[:, i], 6/111)# 7km smoothing
    
                Ta.append(ta)  


                    
            Ta = np.concatenate(Ta, axis = 1) 

            Ta = np.squeeze(Ta)
            Ta_all.append(Ta)
    
    return np.vstack(Ta_all), np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(stype_all)


#%%
#plot PDFs for all 4 channels   
def plot_pdf_gmi(Ta, Tb, bins= None, figname = "distribution_gmi.pdf"):
    
    if bins is None:
        bins = np.arange(100, 310, 2)
        
    fig, axs = plt.subplots(2,2, figsize = [30, 20])
    fig.tight_layout(pad=3.0)
    
    for i, ax in enumerate(fig.axes):
    
        hist_a = np.histogram(Ta[:, i],  bins, density = True)      
        hist_b = np.histogram(Tb[:, i],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0], label =  'observed', linewidth  = 2)       
        ax.plot(bins[:-1], hist_a[0], label =  "simulated", linewidth = 2 )

        ax.set_title(freq[i] + " GHz")
    
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
    
#%%           
    
# def kde(x, y):
        
#     xy = np.vstack([x,y])
#     z = gaussian_kde(xy)(xy)
    
#     return z
#%%
def plot_hist2d(ta, tb0, figname = "hist2d.png"):
    
    pd = ta[:, 0] - ta[:, 1]
    
    pd_sampled = pd[::3]
    ta_sampled = ta[::3, 0]
    
    pd_sampled_gmi = (tb0[:, 0] - tb0[:, 1])
    ta_sampled_gmi = tb0[:, 0]
    
    fig, ax = plt.subplots(1, 1, figsize = [20, 20])
    
    counts,xbins,ybins=np.histogram2d(ta_sampled, pd_sampled, bins=75)
# make the contour plot

    cs = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'blue'))

    
    
        
    counts,xbins,ybins=np.histogram2d(ta_sampled_gmi, pd_sampled_gmi, bins=75)
# make the contour plot

    cs_gmi  = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red'))
    
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
def swap_gmi_183(ta1):
    
        temp = np.zeros(ta1.shape)
        temp[:, 1] = ta1[:, 1]
        temp[:, 0] = ta1[:, 0]
        temp[:, 2] = ta1[:, 3]
        temp[:, 3] = ta1[:, 2]
        ta1 = temp.copy()
        
        return ta1 

#%%
def remove_oversampling_gmi(tb, lat, lon, lsm):
    """
    remove over-sampling in GMI data with latitudes 

    Parameters
    ----------
    tb : np.array, Brightness temp
    lat : np.array, latitudes
    lon : np.array, longitudes
    lsm : np.array, surface class

    Returns
    -------
    tb_sub : np.array, sampled tb
    lat_sub : np.array
    lon_sub : np.array
    lsm_sub : np.array

    """    

    
    fig, ax = plt.subplots(1, 1, figsize = [20, 10])
    
    bins = np.arange(-65, 66, 1)
    hist = np.histogram(lat_gmi.ravel(), bins, density = True) 
    
    ax.hist(lat, bins, density = True)
    
    factors = hist[0]/hist[0].min() 
    
    ilat   = np.digitize(lat, bins)
    icount = np.bincount(ilat)
    
    tb_sub  = []
    lat_sub = []
    lon_sub = []
    lsm_sub = []
    
    for i in range(1, len(bins)):
        
# calculate the oversampling factor for each lat bin
        factor = 1 - 1/factors[i-1]
        
        n = np.int(icount[i] * factor) 
        print (factor, n)
 
        iargs = np.where(ilat == i)[0]
        random.shuffle(iargs)  
        iargs_sub = iargs[n:]
        
        tb_sub.append(tb_gmi[iargs_sub, :])
        lat_sub.append(lat_gmi[iargs_sub])
        lon_sub.append(lon_gmi[iargs_sub])
        lsm_sub.append(lsm_gmi[iargs_sub])
        
   
    
    tb_sub = np.vstack(tb_sub)
    lat_sub = np.concatenate(lat_sub)     
    lon_sub = np.concatenate(lon_sub) 
    lsm_sub = np.concatenate(lsm_sub)


    ax.hist(lat_sub, bins, density = True)   
    ax.set_ylabel("number density")
    ax.set_xlabel("Latitude")
    ax.legend(["original", "sampled"])    

        
        
        
    return tb_sub, lat_sub, lon_sub, lsm_sub            
        
    
#%%
def plot_locations_map(lat0, lon0, z = None):   
    """
    plt lat/lon locations on map

    Parameters
    ----------
    lat0 : np.array, latitudes
    lon0 : np.array, longitudes
    z : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(20, 10))

    m = Basemap(llcrnrlon=0.,llcrnrlat=-85.,urcrnrlon=360.,urcrnrlat=85.,\
                  rsphere=(6378137.00,6356752.3142),\
                  resolution='c',projection='cyl')
#    plt.title(os.path.basename(matfile))    
    m.shadedrelief(scale = 0.1)

    lon0 = lon0 % 360
    cs = m.scatter(lon0, lat0, latlon = True, cmap = "PiYG", vmin = 0, vmax = 300)
    if z is not None:
        cs = (m.scatter(lon0, lat0, latlon = True, c = z, 
                        cmap = "Reds", vmin = 0, vmax = 300))
    plt.colorbar(cs)
    plt.show()  
#    plt.savefig('try.png', bbox_inches = 'tight')        
            
#%%

if __name__ == "__main__":
    
    # GMI satellite data   
    #inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    gmifiles = glob.glob(os.path.join(inpath, "*/*.HDF5"))
    
    random.shuffle(gmifiles)
    gmi_sat = GMI_Sat(gmifiles[:20])
    
    lat_gmi = gmi_sat.lat.values
    lon_gmi = gmi_sat.lon.values
    tb_gmi  = gmi_sat.tb.values
    lsm_gmi = gmi_sat.get_lsm().values
    
    
    lat_gmi = lat_gmi.ravel()
    lon_gmi = lon_gmi.ravel()
    tb_gmi  = tb_gmi.reshape(-1, 4)
    lsm_gmi = lsm_gmi.ravel()
    
    tb_gmi, lat_gmi, lon_gmi, lsm_gmi = remove_oversampling_gmi(tb_gmi, lat_gmi, lon_gmi, lsm_gmi)
    
    # GMI simulations    
    inpath   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.2')  
     
    matfiles = glob.glob(os.path.join(inpath, "2009_3*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath, "2010_*.mat"))
    
    matfiles = matfiles + matfiles1
    
    ta, lat, lon, stype = get_Ta(matfiles)
        
    # GMI frequencies
    freq     = ["166.5V", "166.5H", "183+-3", "183+-7"]
    

    
#%% Tropics
    
    
    print ("doing tropics, all")
    
    latlims  = [0, 30]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = None))
    
#    ta, lat, lon, stype  = get_Ta(matfiles, latlims, lsm = None)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = None)
    
    ta1 = swap_gmi_183(ta1)
    
   

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_tropics_all.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_tropics_all.png")
    
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_tropics_all.png")
   

#%% higher latitudes
    
    print ("doing high lats, all")
    
    latlims  = [0, 65]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = None))
    
#    ta, lat, lon  = get_Ta(matfiles, latlims, lsm = None)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = None)
    
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_highlat_all.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_highlat_all.png")
    
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_highlat_all.png")


#%% tropics land    
    print ("doing tropics, land")
        
    latlims  = [0, 30]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = [3, 4, 5, 6, 7, 13]))
    
#   ta, lat, lon  = get_Ta(matfiles, latlims, lsm = None)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 1)
    
    
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_tropics_land.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_tropics_land.png")    
    
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_tropics_land.png")

#%% higher latitudes land
    
    print ("doing high lats, land")
    
    latlims  = [30, 65]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = [3, 4, 5, 6, 7]))
    
#    ta, lat, lon  = get_Ta(matfiles, latlims, lsm = 1)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 1)    
    
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_highlat_land.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_highlat_land.png")
    plot_hist2d(ta1, tb0,  figname = "hist2d_gmi_highlat_land.png")

    
#%% higher latitudes sea
    
    print ("doing high lats, sea")
    
    latlims  = [30, 65]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = 1))
    
    #ta, lat, lon  = get_Ta(matfiles, latlims, lsm = 0)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 0)    
    
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_highlat_sea.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_highlat_sea.png")
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_highlat_sea.png")
   
    
    
    #%% tropics sea
    
    print ("doing tropics, sea")

    latlims  = [0, 30]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, lsm = 1))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 0)
    
    ta1 = swap_gmi_183(ta1)
    
#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_tropics_sea.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_tropics_sea.png") 
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_tropics_sea.png")

    
#%% higher latitudes sea-ice
    
    print ("highlats, sea-ice")
    
    latlims  = [45, 65]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims , lsm = 2))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 3)
    
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_highlat_seaice.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_highlat_seaice.png")
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_highlat_sea-ice.png")

#%%
    print ("highlats, snow")
    latlims  = [30, 65]
        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims , lsm = [8, 9, 10, 11]))
    
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, lsm = 2)
    ta1 = swap_gmi_183(ta1)

#    plot_scatter(ta, tb0, freq, figname = "scatter_gmi_highlat_snow.png")
    
#    plot_pdf_gmi(ta, tb0, figname = "pdf_gmi_highlat_snow.png")    
    plot_hist2d(ta1, tb0, figname = "hist2d_gmi_highlat_snow.png")
          

#%%
    
    
    m1 = tb0[:, 0] < 300
    m2 = tb0[:, 0] - tb0[:, 1] > 0
    
    m = np.logical_and(m1, m2)
    
    plot_locations_map(lat0[m], lon0[m])
    

    
    
#%%
    
    m1 = ta1[:, 0] > 200
    m2 = ta1[:, 0] - ta1[:, 1] > 15
    
    m = np.logical_and(m1, m2)

    
    plot_locations_map(lat1[m], lon1[m])

    
#%%
    plt.figure(figsize=(12, 6))
    m = Basemap(llcrnrlon=0.,llcrnrlat=-85.,urcrnrlon=360.,urcrnrlat=85.,\
                  rsphere=(6378137.00,6356752.3142),\
                  resolution='c',projection='cyl')
    m.shadedrelief(scale = 0.1)
    
    lon = lon % 360
    m.scatter(lon0,lat0, latlon = True)
    plt.savefig('try.png', bbox_inches = 'tight')
    plt.show()    

#%%

for matfile in matfiles:
    
    print (matfile)

        
 #   gmi = GMI(matfile)
    
    ta, lat, lon = get_Ta([matfile], latlims= [30, 65], lsm = None)
    
    diff = ta[:, 0] - ta[:, 1]
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.hist(diff, np.arange(-60, 60, 1), density = True)
    ax.set_xlabel("PD [V - H]")
    ax.set_ylabel("PDF")
    ax.set_title(os.path.basename(matfile))
    
    plt.figure(figsize=(20, 10))

    m = Basemap(llcrnrlon=0.,llcrnrlat=-85.,urcrnrlon=360.,urcrnrlat=85.,\
                  rsphere=(6378137.00,6356752.3142),\
                  resolution='c',projection='cyl')
#    plt.title(os.path.basename(matfile))    
    m.shadedrelief(scale = 0.1)

    lon = lon % 360
    cs = m.scatter(lon0, lat0, latlon = True, cmap = "PiYG", vmin = -40, vmax = 40)
    plt.colorbar(cs)
    plt.show()  
    plt.savefig('try.png', bbox_inches = 'tight')
    
    
#%%
import matplotlib.pyplot as plt
import scipy.stats as stats

lower, upper = 1, 1.4
mu, sigma = 1.2, 1
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)    

    
    
    
    
    
    
    

    
        