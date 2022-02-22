#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:15:45 2021

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LogNorm

def plot_hist(ta, tb, figname = "contour2d.png", ax = None):

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    
    xdat = ta[:, 0]
    ydat = ta[:, 0] -  ta[:, 1]
    
#    xyrange = [[xdat.min()-5, xdat.max()+5],[ydat.min()-5, ydat.max()+ 5]] # data range
    xyrange = [[100, 300], [-5, 55]] # data range
  
    bins = [100, 60] # number of bins
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
    sigma = 0.1
    hh = gaussian_filter(hh, sigma)
    print (hh.min(), hh.max())
    
    normi= LogNorm(vmin=1e-3, vmax=0.1)
    
    cs = ax.contour(np.flipud(hh.T), colors = "tab:blue",
                    extent=np.array(xyrange).flatten(), 
                #locator= ticker.LogLocator(), 
                #levels =  [0.5e-5, 1e-5, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.05, 0.1],
                levels = 5,
                norm = normi,
                linewidth = 2.5,
                origin='upper')
    
    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()
    ax.clabel(cs, inline=1, inline_spacing = 7,fontsize=18, fmt = fmt)
#    plt.colorbar()   
    ax.plot(xdat1, ydat1, '.',color="tab:blue", alpha = 0.4)
    
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
    
    sigma = 0.1
    hh = gaussian_filter(hh, sigma)
    cs_gmi = ax.contour(np.flipud(hh.T), colors = "tab:red",
                       #levels =  [0.5e-5, 1e-5, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.05, 0.1], 
                       #levels = [0.1]
                       levels = 5,
                       norm = normi,
                       extent=np.array(xyrange).flatten(), 
                  origin='upper')
    ax.clabel(cs_gmi, inline=1, inline_spacing = 7, fontsize=18, fmt = fmt)
 #  plt.colorbar()   
    ax.plot(xdat1, ydat1, '.',color="tab:red",  alpha = 0.4)
    lines = [ cs.collections[0], cs_gmi.collections[0]]
#    labels = ['CS1_neg','CS1_pos','CS2_neg','CS2_pos']
    ax.legend(lines, ["simulated", "observed"], loc = 'upper left')
    #ax.set_xlabel(" Brightness temperature 166 V [K] ")
    #ax.set_ylabel("Polarisation difference [V-H] [K]")
    #ax.set_title("TRO")
    
    # p = [(222,52), (270,52),  (270,16), (222,16),]
    # poly = plt.Polygon(p, ec="tab:gray", fc = None, fill = 0, ls = '--', lw = 2)
    # ax.add_patch(poly)
    # ax.annotate('surface', (265, 54.5),
    #         #xytext=(0.8, 0.9), textcoords='axes fraction',
    #         fontsize=28,
    #         horizontalalignment='right', verticalalignment='top')
    
    # p = [(150,14), (250,14), (250,-2), (150,-2)]
    # poly = plt.Polygon(p, ec="tab:purple", fc = None, fill = 0, ls = '--', lw = 2)
    # ax.add_patch(poly)
    # ax.annotate('cloudy', (172, 16.5),
    #         #xytext=(0.8, 0.9), textcoords='axes fraction',
    #         fontsize=28,
    #         horizontalalignment='right', verticalalignment='top')
    
    
    # p = [(255,6.5), (280,6.5), (280, -3),(255, -3)]
    # poly = plt.Polygon(p, ec="tab:brown", fc = None, fill = 0, ls = '--', lw = 2)
    # ax.add_patch(poly)
    # ax.annotate('clear', (290, 9),
    #         #xytext=(0.8, 0.9), textcoords='axes fraction',
    #         fontsize=28,
    #         horizontalalignment='right', verticalalignment='top')

    #fig.savefig("Figures/" + figname, bbox_inches = "tight")    