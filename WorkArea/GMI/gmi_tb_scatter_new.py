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
import scipy
from matplotlib import ticker, cm
plt.rcParams.update({'font.size': 30})




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
    nedt = np.array([0.70, 0.65, 0.47, 0.56 ])
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
            

            lsm_new = filter_stype(gmi.lat, gmi.stype )
            stype_all.append(lsm_new)
            
            Ta = []
           
            for i in range(4):
            # calculate antenna weighted values
                ta = apply_gaussfilter(gmi.lat, gmi.tb[:, i], 6/111)# 6km smoothing
                ta = add_gaussian_noise(ta, [nedt[i]])
    
                Ta.append(ta)  


                    
            Ta = np.concatenate(Ta, axis = 1) 

            Ta = np.squeeze(Ta)
            Ta_all.append(Ta)
    
    return np.vstack(Ta_all), np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(stype_all)

    
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
        hist_b = np.histogram(Tb[::6, i],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0],'b', label =  freq[i]+ ' obs', linewidth = 2, alpha = 0.5)       
        ax.plot(bins[:-1], hist_a[0],'b--', label =  freq[i] + ' sim', linewidth = 2, alpha = 0.5)
        
        hist_a = np.histogram(Ta[::3, i+1],  bins, density = True)      
        hist_b = np.histogram(Tb[::6, i+1],  bins, density = True)  
        
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
    hist = np.histogram(lat.ravel(), bins, density = True) 
    
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
 
        iargs = np.where(ilat == i)[0]
        random.shuffle(iargs)  
        iargs_sub = iargs[n:]
        
        tb_sub.append(tb[iargs_sub, :])
        lat_sub.append(lat[iargs_sub])
        lon_sub.append(lon[iargs_sub])
        lsm_sub.append(lsm[iargs_sub])
        
   
    
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

    if z is not None:
        cs = (m.scatter(lon0, lat0, latlon = True, c = z, 
                        cmap = "tab20c"))
        
    else:
        cs = m.scatter(lon0, lat0, latlon = True, cmap = "PiYG", vmin = 0, vmax = 300)
    plt.colorbar(cs)
    plt.show()  
#    plt.savefig('try.png', bbox_inches = 'tight')        

#%%
def call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims = None, stype_sim = None, stype_gmi = None, 
                figname = None):

        
    tb0, lat0, lon0 = (filter_gmi_sat(lat_gmi[::6], lon_gmi[::6], 
                                      tb_gmi[::6, :], lsm_gmi[::6],
                                      latlims, stype_gmi))
    
#    ta, lat, lon, stype  = get_Ta(matfiles, latlims, lsm = None)
    ta1, lat1, lon1 = filter_gmi_sat(lat, lon, ta, stype, latlims, stype_sim)
    
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
    

if __name__ == "__main__":
    
    # GMI satellite data   
    #inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    gmifiles = glob.glob(os.path.join(inpath, "*/*.HDF5"))
    
    random.shuffle(gmifiles)
    gmi_sat = GMI_Sat(gmifiles[:10])
    
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
    inpath   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_f07')  
    inpath1   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.2') 
 
     
    matfiles = glob.glob(os.path.join(inpath, "2010_*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath1, "2010_*.mat"))

    
    #matfiles = matfiles + matfiles1 
    ta, lat, lon, stype = get_Ta(matfiles[:])
    ta = swap_gmi_183(ta)
        
    # GMI frequencies
    freq     = ["166.5V", "166.5H", "183+-3", "183+-7"]
    

    
#%% Tropics    
    print ("doing tropics, all")
    latlims  = [0, 30]
    lsm = None    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_tropics_all.png")
 
# higher latitudes
    
    print ("doing 30-45, all")    
    latlims  = [30, 45]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm,  
                figname = "hist2d_gmi_30-45_all.png")

    
    print ("doing 45-60, all")
    
    latlims  = [45, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, lsm, lsm, 
                figname = "hist2d_gmi_45-60_all.png")


#%% tropics land    
    print ("doing tropics, land")
    
    stype_gmi = [3, 4, 5, 6, 7, 13]
    stype_sim = [1, 4, 8]
        
    latlims  = [0, 30]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_tropics_land.png")        

# higher latitudes land
    
    print ("doing high lats, land")
    
    latlims  = [30, 45]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_land.png")
    
    print ("doing high lats, land")
    
    latlims  = [45, 65]    
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-60_land.png")
    
#%% higher latitudes sea
    
    print ("doing high lats, sea")
    
    stype_gmi = [1, 12]
    stype_sim = 0
    
    latlims  = [30, 45]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_sea.png")
    
 
    print ("doing high lats, sea")
    
    latlims  = [45, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-60_sea.png")        
    
    # tropics sea
    
    print ("doing tropics, sea")

    latlims  = [0, 30]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi,  
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_tropics_sea.png")  
    
    
#%% higher latitudes sea-ice
    
    print ("highlats, sea-ice")
    
    stype_gmi = [2,14]
    stype_sim = [3,6]
    
    latlims  = [45, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_highlat_sea-ice.png") 
    
#%%
    print ("highlats, snow")
    stype_gmi = [8, 9, 10, 11]
    stype_sim = [2, 5, 7, 9]
    
    latlims  = [30, 45]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_30-45_snow.png")           
    
    print ("highlats, snow")
    latlims  = [45, 65]
    call_hist2d(ta, lat, lon, stype, tb_gmi, lat_gmi, lon_gmi, 
                lsm_gmi, latlims, stype_sim, stype_gmi, 
                figname = "hist2d_gmi_45-60_snow.png")  

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
    hist = np.histogram(tb_gmi[::6, 0] - tb_gmi[::6, 1], bins, density = True) 
    ax.step(hist[1][:-1], hist[0], 'b', label = "observed")    
    ax.legend()   
    ax.set_yscale('log')
    ax.set_xlabel("Polarisation difference 166 GHz [V-H]")
    ax.set_ylabel("PDF [#/K]")
    fig.savefig("Figures/PD_PDF.pdf", bbox_inches = "tight")
    

#%%  
    latlims = [45, 65]
    stype_sim = [8, 9, 10, 11]
    ta1, lat1, lon1 = filter_gmi_sat(lat_gmi, lon_gmi, tb_gmi, lsm_gmi, latlims, stype_sim)      
    m1 = ta1[:, 0] > 183
    m2 = ta1[:, 0] - ta1[:, 1] > 20
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


#%%
    #histogram definition



    
    
    
    
    
    
    

    
        