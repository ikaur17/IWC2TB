#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:19:44 2021

@author: inderpreet
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def plot_locations_map(lat0, lon0, z = None, maplims = None):   
    """
    plt lat/lon locations on map

    Parameters
    ----------
    lat0 : np.array, latitudes
    lon0 : np.array, longitudes
    z : TYPE, optional
        DESCRIPTION. The default is None.
    maplims = [la1, la2, lo1, lo2]    

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(20, 10))

    if maplims == None:
        m = Basemap(llcrnrlon=0.,llcrnrlat=-85.,urcrnrlon=360.,urcrnrlat=85.,\
                  rsphere=(6378137.00,6356752.3142),\
                  resolution='c',projection='cyl')
    else:
        la1, la2, lo1, lo2 = maplims
    
        m = Basemap(llcrnrlon=lo1,llcrnrlat=la1,urcrnrlon=lo2,urcrnrlat=la2,\
                  rsphere=(6378137.00,6356752.3142),\
                  resolution='c',projection='cyl') 
#    plt.title(os.path.basename(matfile))    
    m.shadedrelief(scale = 0.1)

    lon0 = lon0 % 360

    if z is not None:
        cs = (m.scatter(lon0, lat0, latlon = True, c = z, 
                       # cmap = "tab20c", vmin=np.nanmin(z), vmax=np.nanmax(z)))
                        cmap = "tab20c", vmin=0, vmax=10))        
    else:
        cs = m.scatter(lon0, lat0, latlon = True, cmap = "PiYG")
    plt.colorbar(cs)
    plt.show()  
#    plt.savefig('try.png', bbox_inches = 'tight')   