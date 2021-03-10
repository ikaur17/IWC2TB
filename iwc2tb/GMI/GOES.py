#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:36:26 2021

@author: inderpreet
"""


import numpy as np
import netCDF4
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import numpy.ma as ma
plt.rcParams.update({'font.size': 12})

class GOES():
    
    def __init__(self, filename):
        
            self.dataset       = netCDF4.Dataset(filename)
            self.rad           = self.dataset.variables["Rad"][:]
            self.lat, self.lon = self.get_latlon()
        
    def get_latlon(self):
        
        # GOES-R projection info and retrieving relevant constants
        
        proj_info = self.dataset.variables['goes_imager_projection']
        lon_origin = proj_info.longitude_of_projection_origin
        H = proj_info.perspective_point_height+proj_info.semi_major_axis
        r_eq = proj_info.semi_major_axis
        r_pol = proj_info.semi_minor_axis
    
        # grid info
        lat_rad_1d = self.dataset.variables['x'][:]
        lon_rad_1d = self.dataset.variables['y'][:]
    
        
        # close file when finished
        #dataset.close()    
        # create meshgrid filled with radian angles
        lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)
        
        # lat/lon calc routine from satellite radian angle vectors
        
        lambda_0 = (lon_origin*np.pi)/180.0
    
        a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
        b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
        c_var = (H**2.0)-(r_eq**2.0)
    
        r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    
        s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
        s_y = - r_s*np.sin(lat_rad)
        s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)
    
        lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
        lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)

        return lat, lon
    
    @staticmethod
    def plot_scene(lat, lon, rad):
        
        fig = plt.figure(figsize=(8,6), dpi=200)
        bbox = [np.min(lon),np.min(lat),np.max(lon),np.max(lat)] # set bounds for plotting
        n_add = 0
        m = Basemap(llcrnrlon=bbox[0]-n_add,llcrnrlat=bbox[1]-n_add,
                    urcrnrlon=bbox[2]+n_add,urcrnrlat=bbox[3]+n_add,resolution='l',
                    projection='cyl')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)

        
        m.pcolormesh(lon.data, lat.data, rad, latlon=True)

        #m.pcolormesh(lon, lat, rad, latlon=True)
        
        parallels = np.linspace(np.min(lat),np.max(lat),5)
        m.drawparallels(parallels,labels=[True,False,False,False])
        meridians = np.linspace(np.min(lon),np.max(lon),5)
        m.drawmeridians(meridians,labels=[False,False,False,True])
        cb = m.colorbar()    
    
#%%
    
if __name__ == "__main__":
    
    path = "/home/inderpreet/git/Projects/pansat/notebooks/products/GOES-16/GOES-16-ABI-L1b-RadC"
    file = "OR_ABI-L1b-RadC-M6C13_G16_s20210320716118_e20210320718503_c20210320719001.nc"
    filename = os.path.join(path, file)
    
    goes = GOES(filename)

    lat = goes.lat
    lon = goes.lon
    rad = goes.rad
    
    #lat = lat.data
    lon = lon%360
    #rad = rad.data
    
    GOES.plot_scene(lat, lon, rad)
    
    latmask = (lat > 20) & (lat < 50)
    lonmask = (lon > 175) & (lon < 250)
    
    mask = np.logical_and(latmask, lonmask)
    mask = np.logical_or(rad.mask, ~mask)
    
    rad1 = ma.masked_array(rad.data, mask)
    lat1 = ma.masked_array(lat.data, mask)
    lon1 = ma.masked_array(lon.data, mask)
    
    GOES.plot_scene(lat1, lon1, rad1)