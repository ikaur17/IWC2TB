#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:26:24 2021

@author: inderpreet
"""

import numpy as np
import xarray
from scipy.interpolate import RegularGridInterpolator
import re
import os
import glob
from datetime import datetime

class GMI_Sat():
    
    def __init__(self, filenames):
        
        if np.isscalar(filenames):
#            print ('doing only one file')
            filenames = [filenames]
        self.files = filenames    
            
        # get all corresponding GPROF files:
        gprofiles      = self.get_gprofiles()    
        self.gprofiles = gprofiles
        
        if gprofiles[0] != "noobs_file":
            dataset = xarray.open_dataset(filenames[0], group = "S2")
        
            lat0    = dataset["Latitude"]
            lon0    = dataset["Longitude"]
            tb0     = dataset["Tb"]
        
            dataset.close()
        
        
        if len(filenames) != 1:       

            for i, filename in enumerate(filenames):
                if gprofiles[i] != "noobs_file":
                    print (i)
                
                    dataset = xarray.open_dataset(filename, group = "S2")
            
                    lat    = dataset["Latitude"]
                    lon    = dataset["Longitude"]
                    tb     = dataset["Tb"]
                    
                    lat0 = xarray.concat([lat0, lat], dim = "phony_dim_40")
                    lon0 = xarray.concat([lon0, lon], dim = "phony_dim_40")
                    tb0  = xarray.concat([tb0, tb], dim = "phony_dim_40") 
                
                # lat0 = xarray.concat([lat0, lat], dim = "phony_dim_8")
                # lon0 = xarray.concat([lon0, lon], dim = "phony_dim_8")
                # tb0  = xarray.concat([tb0, tb], dim = "phony_dim_8") 
                
                
       
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
    
    def get_gprofiles(self, gprofpath = None):
        """
        get GPROF file corresponding to the GMI file

        Parameters
        ----------
        gprofpath : list of string, optional
            the path containing gproffiles. The default is None.
            If None, the path on Dendrite is used, 
            otherwise give the path where GPROF data is stored

        Returns
        -------
        gproffiles : list containing the GPROF files

        """
        #1C-R.GPM.GMI.XCAL2016-C.20160131-S185731-E203001.010940.V05A.HDF5
        if gprofpath is None:
            gprofpath = os.path.expanduser('~/Dendrite/SatData/GPROF/')
        gprofiles = []
                
        for file in self.files:
                
            # get date
            #m = re.search('-C.(.+?)-S', file)
            m = re.search('TB2016.(.+?)-S', file)
            date = datetime.strptime(m.group(1) , "%Y%m%d")
            
            # get date and time string
            #m = re.search('-C.(.+?)-E', file)
            m = re.search('-S(.+?)-E', file)
            print (m.group(1))
            
            try:
                gproffile = glob.glob(os.path.join(gprofpath, 
                                               str(date.strftime("%Y")),
                                               str(date.strftime("%m")),
                                               '*' + date.strftime("%Y%m%d") + '-S' +
                                               '*' + m.group(1)+ '*'))
                gprofiles.append(gproffile[0])

            except:
                print ("GPROF data not availble for %s", file) 
                gprofiles.append("noobs_file")
                    
            
        return gprofiles     

    def get_lsm(self): 
    
        
        if self.gprofiles[0] != "noobs_file":

            dataset = xarray.open_dataset(self.gprofiles[0], group = "S1")
            surface0 = dataset["surfaceTypeIndex"]
            dataset.close()

            
        if len(self.gprofiles) > 1:              
           ix = 0
           for gprofile in self.gprofiles:
               
               if gprofile != "noobs_file":
                   print (ix)
                   dataset  = xarray.open_dataset(gprofile, group = "S1")
                   
                   surface  = dataset["surfaceTypeIndex"]
                    
                   surface0 = xarray.concat([surface0, surface], dim = "phony_dim_7")
                   
                   dataset.close()
               
               ix = ix +1   
        self.surface = surface0     
        
        return self.surface
                           

    # def get_data(self, parameter):
    #     """
    #     get data for the parameter 
        
    #     Currently only accessing data from high freq channels under "S2"

    #     Parameters
    #     ----------
    #     parameter : string containing name of the SDS variable

    #     Returns
    #     -------
    #     np.array containing data

    #     """
    #     data = []
    #     for i, file in enumerate(self.files):
    #         print (i)
    #         dataset = xarray.open_dataset(file, group = "S2")
    #         data.append(dataset[parameter].values)
    #         dataset.close()
    #     return  data
    
    # def lsm(self):
    
    #     lsm = xarray.open_dataset("/home/inderpreet/data/land_sea_mask.nc")
    #     lsm = lsm.sortby('latitude' , ascending = True)    
    #     field = lsm.lsm.values[0]
    #     lat   =  lsm.latitude.values
    #     lon   = lsm.longitude.values
        
        
    #     my_interpolating_function = (RegularGridInterpolator((lat, lon), field, 
    #                                               method= "nearest", bounds_error = False, 
    #                                               fill_value = None))     
    #     print("ok")

    #     lsm = np.zeros(lat.shape)
        
    #     self.lon = (self.lon + 180) % 360 - 180
    #     lsm = []
    #     for i in range(self.lat.shape[1]):    
    #         print (i)
            
    #         pts = [[self.lat[j, i], self.lon[j, i]] for j in range(self.lat.shape[0])]  
    #         l = my_interpolating_function(pts)
    #         lsm.append(l)
            
    #     grid_lsm =  np.vstack(lsm).T    

    #     # iland         = grid_lsm > 0.5
    #     # isea          = grid_lsm <= 0.5
        
        
    #     # grid_lsm[iland]  = 1
    #     # grid_lsm[isea]   = 0
            
    #     return grid_lsm
            


               
               
               
                
               
           
            

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