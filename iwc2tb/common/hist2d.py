#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:43:10 2021

@author: inderpreet
"""

import numpy as np


def hist2d(xdat, ydat, xyrange = None, bins = None):
    """
    generates the numerical data for 2d histogram

    Parameters
    ----------
    xdat : input x data
    ydat : input y data
    bins : [n, m], optional
        number of bins wanted in x and y direction. The default is None.

    Returns
    -------
    hh     : bi-dimensional histogram of samples x and y.
    xyrange: [[xdat.min()-1, xdat.max()+1],[ydat.min()-1, ydat.max()+ 1]]
             The range of x and y considered 
    xdat1  : low density points in x 
    ydat1  : low density points in y
    
    To plot the 2d histogram use the returned data as:
        
        fig, ax = plt.subplots(1, 1, figsize = [8, 8])
        cs = ax.contourf(np.flipud(hh.T), cmap= 'Blues',
                extent=np.array(xyrange).flatten(), 
            locator= ticker.LogLocator(), origin='upper')
        cbar = fig.colorbar(cs)  
        ax.plot(xdat1, ydat1, '.',color='blue', alpha = 0.2)

    """
        
    
    #fig, ax = plt.subplots(1, 1, figsize = [12, 12])
    
    if xyrange == None:
        xyrange = [[xdat.min()-1, xdat.max()+1],[ydat.min()-1, ydat.max()+ 1]] # data range

  
    if bins == None:
        bins = [100, 100] # number of bins
   
    thresh = 1/xdat.shape[0] * 2  #density threshold
    
    
    # histogram the data
    hh, locx, locy = np.histogram2d(xdat, ydat, 
                                    range=xyrange, bins=bins, density = True)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    
    return hh, xyrange, xdat1, ydat1
