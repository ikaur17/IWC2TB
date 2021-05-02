#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:15:45 2021

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def plot_hist(ta, tb, figname = "contour2d.png"):

    
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