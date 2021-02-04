#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:26:24 2021

@author: inderpreet
"""

import numpy as np
import xarray
from scipy.interpolate import RegularGridInterpolator

class GMI_Sat():
    
    def __init__(self, filenames):
        
        if np.isscalar(filenames):
#            print ('doing only one file')
            filenames = [filenames]
            
        dataset = xarray.open_dataset(filenames[0], group = "S2")
        
        lat0    = dataset["Latitude"]
        lon0    = dataset["Longitude"]
        tb0     = dataset["Tb"]
        
        dataset.close()
        
        self.files = filenames
        
        if len(filenames) != 1:       

            for i, filename in enumerate(filenames):
                print (i)
                
                dataset = xarray.open_dataset(filename, group = "S2")
        
                lat    = dataset["Latitude"]
                lon    = dataset["Longitude"]
                tb     = dataset["Tb"]
                
                lat0 = xarray.concat([lat0, lat], dim = "phony_dim_40")
                lon0 = xarray.concat([lon0, lon], dim = "phony_dim_40")
                tb0  = xarray.concat([tb0, tb], dim = "phony_dim_40") 
                
       
        self.lat = lat0
        self.lon = lon0
        self.tb  = tb0 

        dataset.close()        
           
            
#            self.file.append(h5py.File(filename, 'r'))
#            self.file.append(xarray.open_dataset(filename, group = 'S2'))
        
        
    def get_keys(self) :
        """
        get SDS keys

        Returns
        -------
        list containing names of the SDS names

        """
        #Names of the groups in HDF5 file.
        
        return self.file[0].keys()

    def get_data(self, parameter):
        """
        get data for the parameter 
        
        Currently only accessing data from high freq channels under "S2"

        Parameters
        ----------
        parameter : string containing name of the SDS variable

        Returns
        -------
        np.array containing data

        """
        data = []
        for i, file in enumerate(self.files):
            print (i)
            dataset = xarray.open_dataset(file, group = "S2")
            data.append(dataset[parameter].values)
            dataset.close()
        return  data
    
    def lsm(self):
    
        lsm = xarray.open_dataset("/home/inderpreet/data/land_sea_mask.nc")
        lsm = lsm.sortby('latitude' , ascending = True)    
        field = lsm.lsm.values[0]
        lat   =  lsm.latitude.values
        lon   = lsm.longitude.values
        
        
        my_interpolating_function = (RegularGridInterpolator((lat, lon), field, 
                                                  method= "nearest", bounds_error = False, 
                                                  fill_value = None))     
        

        lsm = np.zeros(lat.shape)
        
        self.lon = (self.lon + 180) % 360 - 180
        lsm = []
        for i in range(self.lat.shape[1]):    
            
            pts = [[self.lat[j, i], self.lon[j, i]] for j in range(self.lat.shape[0])]  
            l = my_interpolating_function(pts)
            lsm.append(l)
            
        grid_lsm =  np.vstack(lsm).T    

        iland         = grid_lsm > 0.5
        isea          = grid_lsm <= 0.5
        
        
        grid_lsm[iland]  = 1
        grid_lsm[isea]   = 0
            
        return grid_lsm
            
            

    # @property
    # def lat(self):    
    #     """
    #     Latitude values

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """
        
    #     lat = self.get_data('Latitude')
    #     return np.concatenate(lat, axis = 0)
    
    # @property
    # def lon(self): 
    #     """
    #     Longitude values

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """
        
    #     return self.get_data('Longitude')
    
    # @property
    # def tb(self): 
    #     """
    #     brightness temperature values

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """
        
    #     tb = self.get_data('Tb')
    #     return np.concatenate(tb, axis  = 0)