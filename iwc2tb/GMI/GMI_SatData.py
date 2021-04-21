#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:26:24 2021

Class file to handle GMI L1C and L1B data
can read in one file or multiple files at one time.

@author: inderpreet
"""

import numpy as np
import xarray
import re
import os
import glob
from datetime import datetime, timedelta
from iwc2tb.common.plot_locations_map import plot_locations_map

class GMI_Sat():
    
    def __init__(self, filenames):
        
                                            
        if np.isscalar(filenames):
#            print ('doing only one file')
            filenames = [filenames]
        self.files = filenames    
        
        self.level = os.path.basename(self.files[0])[0:2]
            
        # get all corresponding GPROF files:
        gprofiles      = self.get_gprofiles()   
        self.gprofiles = gprofiles
        
        ix = np.where(np.array(gprofiles, copy=False) != "noobs_file")[0]
        
        if len(ix) == 0:
            raise Exception("Check another file, GPROF unavailable")
            
        # open first file to initialise
        dataset = xarray.open_dataset(filenames[ix[0]], group = "S2") 
        sctime0  = xarray.open_dataset(filenames[ix[0]], group = "S2/ScanTime")
        
        lat0    = dataset["Latitude"]
        lon0    = dataset["Longitude"]
        
        if self.level == "1C":
            var = "Tc"
        if self.level == "1B"     :
            var = "Tb"
            
        tb0     = dataset[var]     
        dataset.close()
        
        # corresponding GPROF dataset
        dataset               = xarray.open_dataset(self.gprofiles[ix[0]], group = "S1")
        self.gprof_parameters = list(dataset.keys())
        dataset.close()
        
        
        # loop over all input GMI files if len(gmifiles) > 1
        if len(filenames) != 1:       

            for i in ix[1:]:
                filename = filenames[i]
            
                dataset = xarray.open_dataset(filename, group = "S2")
        
                lat    = dataset["Latitude"]
                lon    = dataset["Longitude"]
                tb     = dataset[var]    
                sctime = xarray.open_dataset(filename, group = "S2/ScanTime")
                
                if self.level == "1B":
                    lat0    = xarray.concat([lat0, lat], dim = "phony_dim_40")
                    lon0    = xarray.concat([lon0, lon], dim = "phony_dim_40")
                    tb0     = xarray.concat([tb0, tb], dim = "phony_dim_40") 
                    sctime0 = xarray.concat([sctime0, sctime], dim = "phony_dim_23")
                if self.level == "1C":
                    lat0    = xarray.concat([lat0, lat], dim = "phony_dim_8")
                    lon0    = xarray.concat([lon0, lon], dim = "phony_dim_8")
                    tb0     = xarray.concat([tb0, tb], dim = "phony_dim_8")
                    sctime0 = xarray.concat([sctime0, sctime], dim = "phony_dim_6")
                    
                    
                
       
        self.lat = lat0.values
        self.lon = lon0.values
        self.sctime = sctime0
        
        sctime0.close()
        # tbs are in order GMI channels 10, 11, 12, 13
        self.tb  = tb0.values 

        self.time = self.get_pixtime()
        dataset.close()        
        
    
    def get_pixtime(self):
        

        year    = self.sctime["Year"].data
        mon     = self.sctime["Month"].data
        day     = self.sctime["DayOfMonth"].data.astype('timedelta64[D]').astype(np.int32)
        hour    = self.sctime["Hour"].data.astype('timedelta64[h]').astype(np.int32)
        minute  = self.sctime["Minute"].data.astype('timedelta64[m]').astype(np.int32)
        sec     = self.sctime["Second"].data.astype('timedelta64[s]').astype(np.int32)
        
        date    = [datetime(year[i], mon[i], day[i], hour[i], minute[i], sec[i]) for i in range(len(year))]
        
        return np.array(date)
        
    @property
    def lst(self):
        
        t    = self.time.reshape(-1, 1)
        t  = np.tile(t, self.lon.shape[1])
        mins = self.lon * 4.0
        
        lst = t.copy()
        nx = self.lon.shape[0]
        ny = self.lat.shape[1]
        lst = [[t[i, j] + timedelta(minutes = np.float(mins[i, j])) for i in range(nx)] for j in range(ny)]
        
        return np.stack(lst).T
        
    @property
    def t0(self):
        t0 = self.get_gprofdata("temp2mIndex")
        return t0

    @property
    def iwp(self):
        iwp = self.get_gprofdata("iceWaterPath")
        return iwp

    @property
    def rwp(self):
        iwp = self.get_gprofdata("rainWaterPath")
        return iwp        
    
    @property
    def wvp(self):
        wvp = self.get_gprofdata("totalColumnWaterVaporIndex")
        return wvp     
    
    
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
                
            if self.level == "1B":
                m  = re.search('TB2016.(.+?)-S', file)
                m1 = re.search('-S(.+?)-E', file)

            if self.level == "1C":
                m  = re.search('XCAL2016-C.(.+?)-S', file)
                m1 = re.search('-S.(.+?)-E', file)

            
            date = datetime.strptime(m.group(1) , "%Y%m%d")
            
            try:
                gproffile = glob.glob(os.path.join(gprofpath, 
                                               str(date.strftime("%Y")),
                                               str(date.strftime("%m")),
                                               '*' + date.strftime("%Y%m%d") + '-S' +
                                               '*' + m1.group(1)+ '*'))
                

                gprofiles.append(gproffile[0])

            except:
                print ("GPROF data not availble for ", file) 
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
