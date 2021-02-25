#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:26:24 2021

@author: inderpreet
"""

import numpy as np
import xarray
import re
import os
import glob
from datetime import datetime
from iwc2tb.common.plot_locations_map import plot_locations_map

class GMI_Sat():
    
    def __init__(self, filenames):
        
                                            
        if np.isscalar(filenames):
#            print ('doing only one file')
            filenames = [filenames]
        self.files = filenames    
            
        # get all corresponding GPROF files:
        gprofiles      = self.get_gprofiles()    
        self.gprofiles = gprofiles
        
        ix = np.where(np.array(gprofiles, copy=False) != "noobs_file")[0]
        print (ix)

        dataset = xarray.open_dataset(filenames[ix[0]], group = "S2")
    
        lat0    = dataset["Latitude"]
        lon0    = dataset["Longitude"]
        tb0     = dataset["Tb"]
    
        dataset.close()
        
        dataset               = xarray.open_dataset(self.gprofiles[ix[0]], group = "S1")
        self.gprof_parameters = list(dataset.keys())
        dataset.close()
        
        
        if len(filenames) != 1:       

            for i in ix[1:]:
                filename = filenames[i]

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
                
                
       
        self.lat = lat0.values
        self.lon = lon0.values
        self.tb  = tb0.values 

        dataset.close()        
        
        
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
        
        idx = np.where(np.array(self.gprofiles, copy=False) != "noobs_file")[0]


        dataset = xarray.open_dataset(self.gprofiles[idx[0]], group = "S1")
        surface0 = dataset["surfaceTypeIndex"]
        dataset.close()

            
        if len(self.gprofiles) > 1:              
           ix = 0
           for ix in idx[1:]:
               print (ix)
               gprofile = self.gprofiles[ix]
               

               dataset  = xarray.open_dataset(gprofile, group = "S1")
                
               surface  = dataset["surfaceTypeIndex"]
                 
               surface0 = xarray.concat([surface0, surface], dim = "phony_dim_7")
                
               dataset.close()
 
        self.surface = surface0     
        
        return self.surface.values
                           
    def get_gprofdata(self, parameter): 
        
    
        if parameter in self.gprof_parameters:
            idx = np.where(np.array(self.gprofiles, copy=False) != "noobs_file")[0]
    
    
            dataset = xarray.open_dataset(self.gprofiles[idx[0]], group = "S1")
            data0 = dataset[parameter]
            dataset.close()
    
                
            if len(self.gprofiles) > 1:              
               ix = 0
               for ix in idx[1:]:
                   print (ix)
                   gprofile = self.gprofiles[ix]
                   
    
                   dataset  = xarray.open_dataset(gprofile, group = "S1")
                    
                   data  = dataset[parameter]
                     
                   data0 = xarray.concat([data0, data], dim = "phony_dim_7")
                    
                   dataset.close()
    
            
            return data0.values
        else:
            raise Exception("Parameter should be one of  ", self.gprof_parameters)
    
    
    def plot_scene(self, z = None):        
        """
        plots the overpass of DARDAR

        Returns
        -------
        None.

        """
        plot_locations_map(self.lat, self.lon, z)
